import asyncio
import time

from miner_utils import get_logger

from pearl_gateway.comm.dataclasses import BlockTemplate, MiningJob, MiningPausedError

logger = get_logger(__name__)


class WorkCache:
    """
    Caches the latest block template and provides mining jobs to miners.
    Acts as an in-memory store between the Pearl node and the miner.
    """

    INCOMPLETE_HEADER_BYTES = 76

    def __init__(self):
        self.current_template: BlockTemplate | None = None
        self.last_update_time: float = 0
        self.lock = asyncio.Lock()  # For thread-safe access to template
        self._variants_by_worker_id: dict[int, BlockTemplate] = {}
        self._templates_by_header: dict[bytes, BlockTemplate] = {}

    @staticmethod
    def _validate_worker_id(worker_id: int) -> None:
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not 0 <= worker_id <= 0xFF
        ):
            raise ValueError("worker_id must be an integer from 0 to 255")

    def _cache_variant(self, template: BlockTemplate) -> None:
        header = template.header.serialize_without_proof_commitment()
        if len(header) != self.INCOMPLETE_HEADER_BYTES:
            raise ValueError(
                "incomplete block header must be exactly "
                f"{self.INCOMPLETE_HEADER_BYTES} bytes"
            )
        self._variants_by_worker_id[template.worker_id] = template
        self._templates_by_header[header] = template

    def _get_variant(self, worker_id: int) -> BlockTemplate:
        self._validate_worker_id(worker_id)
        if self.current_template is None:
            raise MiningPausedError("no block template available")

        variant = self._variants_by_worker_id.get(worker_id)
        if variant is None:
            variant = self.current_template.for_worker_id(worker_id)
            self._cache_variant(variant)
        return variant

    async def update_template(self, template: BlockTemplate) -> bool:
        """
        Update the cached block template if it's different from current.
        Returns True if template was updated, False if unchanged.
        """
        async with self.lock:
            is_new = (
                self.current_template is None
                or template.header.previous_block_hash
                != self.current_template.header.previous_block_hash
            )

            if is_new:
                logger.info(f"Updating block template to height {template.height}")
                self.current_template = template
                self._variants_by_worker_id.clear()
                self._templates_by_header.clear()
                self._cache_variant(template)
                self.last_update_time = time.time()
                return True
            else:
                age = time.time() - self.last_update_time
                logger.debug(f"Template unchanged (age: {age:.2f}s)")
                return False

    async def get_mining_job(
        self, worker_id: int = 0, *, include_worker_recipe: bool = False
    ) -> MiningJob:
        """
        Get current mining job for a miner.
        Raises MiningPausedError if no valid template is available.
        """
        async with self.lock:
            try:
                template = self._get_variant(worker_id)
            except MiningPausedError:
                logger.warning("No block template available")
                raise
            return MiningJob.from_template(
                template=template, include_worker_recipe=include_worker_recipe
            )

    async def get_template_for_header(
        self, header: bytes, worker_id: int | None = None
    ) -> BlockTemplate | None:
        """Return the live template whose incomplete header exactly matches ``header``.

        If the exact header is no longer cached, regenerate only the requested worker
        variant from the current base template and accept it only when its serialized
        header is identical to ``header``.
        """
        if not isinstance(header, bytes) or len(header) != self.INCOMPLETE_HEADER_BYTES:
            return None
        async with self.lock:
            template = self._templates_by_header.get(header)
            if template is not None or worker_id is None or self.current_template is None:
                return template

            try:
                variant = self._get_variant(worker_id)
            except (MiningPausedError, ValueError):
                return None

            if variant.header.serialize_without_proof_commitment() != header:
                return None
            return variant

    async def get_template_age(self) -> float | None:
        """Get the age of the current template in seconds."""
        async with self.lock:
            if self.current_template is None:
                return None
            return time.time() - self.last_update_time

    async def invalidate(self) -> None:
        """Invalidate the current template, forcing a refresh on next request."""
        async with self.lock:
            logger.info("Invalidating current template")
            self.current_template = None
            self._variants_by_worker_id.clear()
            self._templates_by_header.clear()
            self.last_update_time = 0
