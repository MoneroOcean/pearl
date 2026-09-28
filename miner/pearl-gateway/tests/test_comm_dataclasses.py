import pytest
from bitcoinutils.transactions import Transaction
from pearl_gateway.blockchain_utils.blockchain_utils import double_sha256
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import (
    BlockTemplate,
    MiningJob,
    b64_decode,
    b64_encode,
)
from pearl_gateway.comm.mining_configuration import PearlMiningConfigurationFactory
from pearl_mining import PENALTY_BASE_RANK
from pydantic import ValidationError


class TestMiningJob:
    """Test MiningJob data structure."""

    def test_mining_job_to_dict(self, sample_block_template):
        """Test MiningJob.to_dict() method."""
        job = MiningJob.from_template(sample_block_template)

        result = job.to_dict()
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()

        expected_keys = {
            "incomplete_header_bytes",
            "target",
            "target_decimal",
            "cert_version",
            "expected_reward",
            "worker_id",
        }
        assert set(result.keys()) == expected_keys
        assert result["target_decimal"] == str(sample_block_template.target)
        assert result["expected_reward"] == sample_block_template.coinbase_value

        assert b64_decode(result["incomplete_header_bytes"]) == expected_header_bytes
        assert result["target"] == sample_block_template.target
        assert result["cert_version"] == int(sample_block_template.required_cert_version)
        assert result["worker_id"] == sample_block_template.worker_id

        # Verify all values are JSON-serializable types
        assert isinstance(result["incomplete_header_bytes"], str)
        assert isinstance(result["target"], int)
        assert isinstance(result["cert_version"], int)

    def test_mining_job_from_dict(self, sample_block_template):
        """Test MiningJob.from_dict() method."""
        expected_header_bytes = sample_block_template.header.serialize_without_proof_commitment()
        data = {
            "incomplete_header_bytes": b64_encode(expected_header_bytes),
            "target": sample_block_template.target,
            "cert_version": int(sample_block_template.required_cert_version),
        }

        job = MiningJob.from_dict(data)

        assert job.incomplete_header_bytes == expected_header_bytes
        assert job.target == data["target"]
        assert job.cert_version == sample_block_template.required_cert_version
        assert job.worker_id is None

    def test_mining_job_round_trip(self, sample_block_template):
        """Test MiningJob to_dict -> from_dict round trip."""
        original_job = MiningJob.from_template(sample_block_template)

        data = original_job.to_dict()
        restored_job = MiningJob.from_dict(data)

        assert restored_job.incomplete_header_bytes == original_job.incomplete_header_bytes
        assert restored_job.target == original_job.target
        assert restored_job.cert_version == original_job.cert_version
        assert restored_job.worker_id == original_job.worker_id
        # Verify complete equality
        assert restored_job == original_job

    def test_mining_job_from_template(self, sample_block_template):
        """Test MiningJob.from_template() method."""
        job = MiningJob.from_template(sample_block_template)

        assert (
            job.incomplete_header_bytes
            == sample_block_template.header.serialize_without_proof_commitment()
        )
        assert job.target == sample_block_template.target
        assert job.cert_version == sample_block_template.required_cert_version
        assert job.worker_id == sample_block_template.worker_id

    def test_mining_job_worker_recipe_response(self, sample_block_template):
        job = MiningJob.from_template(sample_block_template, include_worker_recipe=True)
        data = job.to_dict(include_worker_recipe=True)
        restored_job = MiningJob.from_dict(data)

        assert {
            "worker_coinbase_bytes",
            "worker_coinbase_offset",
            "worker_merkle_branch",
        }.issubset(data)
        assert restored_job.incomplete_header_bytes == job.incomplete_header_bytes
        assert restored_job.worker_id == job.worker_id

    def test_mining_job_from_dict_rejects_short_header(self, sample_block_template):
        data = MiningJob.from_template(sample_block_template).to_dict()
        data["incomplete_header_bytes"] = b64_encode(b"short")

        with pytest.raises(ValueError, match="header must be exactly 76 bytes"):
            MiningJob.from_dict(data)

    def test_worker_recipe_requires_getblocktemplate_source(
        self, sample_block_template, monkeypatch
    ):
        monkeypatch.setattr(sample_block_template, "source_data", None)

        with pytest.raises(ValueError, match="live getblocktemplate source"):
            MiningJob.from_template(sample_block_template, include_worker_recipe=True)


class TestAdjustTarget:
    """The rank penalty applied when turning a block target into a mining target."""

    ROW_INDICES = [0, 8, 64, 72]
    COL_INDICES = [0, 1, 8, 9, 32, 33, 40, 41]
    BLOCK_TARGET = 2**64
    # Valid for every rank exercised here: 16 * rank <= COMMON_DIM <= 4 * rank**2.
    COMMON_DIM = 8192

    def _mining_config(self, rank: int):
        return PearlMiningConfigurationFactory.create(
            common_dim=self.COMMON_DIM,
            rank=rank,
            row_indices=self.ROW_INDICES,
            col_indices=self.COL_INDICES,
        )

    def _mining_job(self, target: int) -> MiningJob:
        return MiningJob(
            incomplete_header_bytes=b"",
            target=target,
            cert_version=CertificateVersion.ZK_MOE,
        )

    def _adjusted_target(self, rank: int) -> int:
        return self._mining_job(self.BLOCK_TARGET).adjust_target(self._mining_config(rank))

    def test_base_rank_is_unpenalized(self):
        """A miner at the base rank searches the plain hash-tile-scaled target."""
        config = self._mining_config(PENALTY_BASE_RANK)
        expected = (
            self.BLOCK_TARGET * config.hash_tile_h * config.hash_tile_w * config.rounded_common_dim
        )
        assert self._adjusted_target(PENALTY_BASE_RANK) == expected

    def test_larger_rank_gets_a_proportionally_smaller_target(self):
        """Doubling the rank halves the target, cancelling the nesting advantage."""
        base = self._adjusted_target(PENALTY_BASE_RANK)
        for multiple in (2, 4):
            adjusted = self._adjusted_target(PENALTY_BASE_RANK * multiple)
            assert adjusted == base // multiple

    def test_rank_below_base_is_rejected(self):
        with pytest.raises(ValueError, match="below the minimum"):
            self._adjusted_target(PENALTY_BASE_RANK // 2)

    def test_degenerate_config_is_rejected(self):
        """A config whose common_dim is below the rank yields a zero adjustment
        factor; adjust_target must surface that as a ValueError rather than
        returning a zero bound."""
        config = PearlMiningConfigurationFactory.create(
            common_dim=PENALTY_BASE_RANK // 2,
            rank=PENALTY_BASE_RANK,
            row_indices=self.ROW_INDICES,
            col_indices=self.COL_INDICES,
        )
        with pytest.raises(ValueError, match="degenerate"):
            self._mining_job(self.BLOCK_TARGET).adjust_target(config)

    def test_target_too_easy_is_rejected(self):
        """A target whose penalized bound exceeds 256 bits must raise. Clamping it
        to the maximum target would have the miner search a target every hash
        satisfies."""
        with pytest.raises(ValueError, match="too easy"):
            self._mining_job(2**240).adjust_target(self._mining_config(PENALTY_BASE_RANK))


class TestBlockTemplateCertVersion:
    """Test that BlockTemplate surfaces the node's required certificate version."""

    def test_required_cert_version_parsed_from_template(
        self, sample_block_template_data, mining_address
    ):
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        for version in CertificateVersion:
            data = {**sample_block_template_data, "requiredcertversion": int(version)}
            template = BlockTemplate.from_get_block_template(
                GetBlockTemplateResponse.model_validate(data),
                mining_address=mining_address,
            )
            assert template.required_cert_version == version

    def test_missing_required_cert_version_defaults_to_v1(
        self, sample_block_template_data, mining_address
    ):
        """An old node that omits requiredcertversion is treated as V1-only."""
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        data = {
            key: value
            for key, value in sample_block_template_data.items()
            if key != "requiredcertversion"
        }
        template = BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(data),
            mining_address=mining_address,
        )
        assert template.required_cert_version == CertificateVersion.ZK_DENSE

    def test_unknown_required_cert_version_is_rejected(self, sample_block_template_data):
        """A version this build has no derivation for must fail loudly rather than
        be mined under the wrong one."""
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        unknown = max(CertificateVersion) + 1
        data = {**sample_block_template_data, "requiredcertversion": unknown}
        with pytest.raises(ValidationError):
            GetBlockTemplateResponse.model_validate(data)


class TestBlockTemplateWorkerVariants:
    """Worker namespaces alter only the coinbase-dependent header fields."""

    @pytest.mark.parametrize("worker_id", [-1, 256, False, True])
    def test_worker_id_boundaries_are_rejected(self, sample_block_template, worker_id):
        with pytest.raises(ValueError, match="worker_id"):
            sample_block_template.for_worker_id(worker_id)

    def test_worker_variants_share_regular_transactions(
        self, sample_block_template
    ):
        variant = sample_block_template.for_worker_id(1)

        assert variant.raw_transactions is sample_block_template.raw_transactions
        assert variant.source_data is sample_block_template.source_data
        assert variant.worker_id == 1
        assert variant.header.serialize_without_proof_commitment() != (
            sample_block_template.header.serialize_without_proof_commitment()
        )

    @pytest.mark.parametrize(
        "default_witness_commitment", [None, "6a24aa21a9ed" + "00" * 32]
    )
    @pytest.mark.parametrize("worker_id", [0, 1, 255])
    def test_worker_recipe_derives_existing_headers(
        self,
        sample_block_template_data,
        mining_address,
        default_witness_commitment,
        worker_id,
    ):
        from pearl_gateway.rpc_types import GetBlockTemplateResponse

        template_data = {
            **sample_block_template_data,
            "default_witness_commitment": default_witness_commitment,
        }
        template = BlockTemplate.from_get_block_template(
            GetBlockTemplateResponse.model_validate(template_data),
            mining_address=mining_address,
        )
        coinbase_bytes, worker_offset, raw_branch = (
            template.get_worker_derivation_recipe()
        )

        assert len(template.header.serialize_without_proof_commitment()) == 76
        assert 0 <= worker_offset < len(coinbase_bytes)
        assert coinbase_bytes[worker_offset] == 0
        assert len(raw_branch) % 32 == 0
        # The fixture has two regular transactions, so this covers the odd
        # three-leaf tree and its duplicated final node.
        assert len(raw_branch) == 64

        derived_coinbase_bytes = bytearray(coinbase_bytes)
        derived_coinbase_bytes[worker_offset] = worker_id
        derived_coinbase = Transaction.from_raw(bytes(derived_coinbase_bytes))
        current_hash = bytes.fromhex(derived_coinbase.get_txid())[::-1]
        for branch_offset in range(0, len(raw_branch), 32):
            current_hash = double_sha256(
                current_hash + raw_branch[branch_offset : branch_offset + 32]
            )
        derived_merkle_root = current_hash[::-1]

        variant = template.for_worker_id(worker_id)
        base_header = template.header.serialize_without_proof_commitment()
        # Header serialization stores hashes in internal little-endian order.
        derived_header = base_header[:36] + derived_merkle_root[::-1] + base_header[68:]
        assert derived_header == variant.header.serialize_without_proof_commitment()
        assert bytes(derived_coinbase_bytes) == variant.coinbase_tx.to_bytes(False)

        if default_witness_commitment is not None:
            assert variant.coinbase_tx.to_bytes(True)[4:6] == b"\x00\x01"
        else:
            assert variant.coinbase_tx.to_bytes(False)[4:6] != b"\x00\x01"

    def test_worker_recipe_rejects_malformed_header_size(
        self, sample_block_template, monkeypatch
    ):
        monkeypatch.setattr(
            sample_block_template.header,
            "serialize_without_proof_commitment",
            lambda: b"",
        )
        with pytest.raises(ValueError, match="header must be exactly 76 bytes"):
            sample_block_template.get_worker_derivation_recipe()
