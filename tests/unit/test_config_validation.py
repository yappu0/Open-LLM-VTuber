"""Tests for configuration validation functionality."""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from src.open_llm_vtuber.config_manager import (
    read_yaml,
    validate_config,
    Config,
    SystemConfig,
    CharacterConfig,
)


class TestReadYaml:
    """Tests for the read_yaml function."""

    def test_read_valid_yaml(self, configs_dir: Path):
        """Test reading a valid YAML configuration file."""
        config_path = configs_dir / "valid_config.yaml"
        config_data = read_yaml(str(config_path))

        assert config_data is not None
        assert "system_config" in config_data
        assert "character_config" in config_data
        assert config_data["system_config"]["host"] == "localhost"
        assert config_data["system_config"]["port"] == 12393

    def test_read_yaml_with_env_vars(self, configs_dir: Path):
        """Test reading YAML with environment variable substitution."""
        # Create a temporary config file with env var placeholder
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_file.write("test_key: '${TEST_ENV_VAR}'\n")
            temp_file.write("other_key: 'normal_value'\n")
            temp_path = temp_file.name

        try:
            # Set environment variable
            os.environ["TEST_ENV_VAR"] = "env_value"

            config_data = read_yaml(temp_path)

            assert config_data["test_key"] == "env_value"
            assert config_data["other_key"] == "normal_value"
        finally:
            # Clean up
            os.unlink(temp_path)
            del os.environ["TEST_ENV_VAR"]

    def test_read_yaml_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            read_yaml("nonexistent_config.yaml")

    def test_read_yaml_preserves_structure(self, configs_dir: Path):
        """Test that nested YAML structure is preserved."""
        config_path = configs_dir / "valid_config.yaml"
        config_data = read_yaml(str(config_path))

        # Check nested structure
        assert "agent_config" in config_data["character_config"]
        assert "llm_configs" in config_data["character_config"]["agent_config"]
        assert (
            "ollama_llm"
            in config_data["character_config"]["agent_config"]["llm_configs"]
        )


class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_validate_minimal_config(self, minimal_config_dict: Dict[str, Any]):
        """Test validating a minimal valid configuration."""
        config = validate_config(minimal_config_dict)

        assert isinstance(config, Config)
        assert isinstance(config.system_config, SystemConfig)
        assert isinstance(config.character_config, CharacterConfig)

    def test_validate_full_config(self, full_config_dict: Dict[str, Any]):
        """Test validating a full configuration with all optional fields."""
        config = validate_config(full_config_dict)

        assert isinstance(config, Config)
        assert config.system_config.host == "localhost"
        assert config.character_config.conf_name == "test_character"

    def test_validate_missing_required_field(self, minimal_config_dict: Dict[str, Any]):
        """Test that validation fails when required field is missing."""
        # Remove required field
        del minimal_config_dict["character_config"]["persona_prompt"]

        with pytest.raises(ValidationError):
            validate_config(minimal_config_dict)

    def test_validate_missing_character_config(
        self, minimal_system_config: Dict[str, Any]
    ):
        """Test that validation fails when character_config is missing."""
        config_data = {"system_config": minimal_system_config}

        with pytest.raises(ValidationError):
            validate_config(config_data)

    def test_validate_invalid_port(self, minimal_config_dict: Dict[str, Any]):
        """Test that validation fails for invalid port number."""
        minimal_config_dict["system_config"]["port"] = 70000  # Invalid port

        with pytest.raises(ValidationError):
            validate_config(minimal_config_dict)

    def test_validate_empty_persona_prompt(self, minimal_config_dict: Dict[str, Any]):
        """Test that validation fails for empty persona_prompt."""
        minimal_config_dict["character_config"]["persona_prompt"] = ""

        with pytest.raises(ValidationError):
            validate_config(minimal_config_dict)


class TestSystemConfig:
    """Tests for SystemConfig validation."""

    def test_valid_system_config(self, minimal_system_config: Dict[str, Any]):
        """Test creating a valid SystemConfig."""
        config = SystemConfig(**minimal_system_config)

        assert config.host == "localhost"
        assert config.port == 12393
        assert config.config_alts_dir == "characters"

    def test_port_boundary_values(self, minimal_system_config: Dict[str, Any]):
        """Test port boundary values."""
        # Valid minimum port
        minimal_system_config["port"] = 0
        config = SystemConfig(**minimal_system_config)
        assert config.port == 0

        # Valid maximum port
        minimal_system_config["port"] = 65535
        config = SystemConfig(**minimal_system_config)
        assert config.port == 65535

    def test_invalid_port_negative(self, minimal_system_config: Dict[str, Any]):
        """Test that negative port raises error."""
        minimal_system_config["port"] = -1

        with pytest.raises(ValidationError):
            SystemConfig(**minimal_system_config)

    def test_invalid_port_too_large(self, minimal_system_config: Dict[str, Any]):
        """Test that port > 65535 raises error."""
        minimal_system_config["port"] = 65536

        with pytest.raises(ValidationError):
            SystemConfig(**minimal_system_config)


class TestCharacterConfig:
    """Tests for CharacterConfig validation."""

    def test_valid_character_config(self, minimal_character_config: Dict[str, Any]):
        """Test creating a valid CharacterConfig."""
        config = CharacterConfig(**minimal_character_config)

        assert config.conf_name == "test_character"
        assert config.conf_uid == "test_character_001"
        assert config.live2d_model_name == "test_model"

    def test_persona_prompt_cannot_be_empty(
        self, minimal_character_config: Dict[str, Any]
    ):
        """Test that empty persona_prompt raises error."""
        minimal_character_config["persona_prompt"] = ""

        with pytest.raises(ValidationError):
            CharacterConfig(**minimal_character_config)

    def test_default_human_name(self, minimal_character_config: Dict[str, Any]):
        """Test that human_name has a default value."""
        del minimal_character_config["human_name"]
        config = CharacterConfig(**minimal_character_config)

        assert config.human_name == "Human"


class TestConfigFromYamlFile:
    """Integration tests for loading and validating config from YAML files."""

    def test_load_and_validate_valid_config(self, configs_dir: Path):
        """Test loading and validating a valid configuration file."""
        config_path = configs_dir / "valid_config.yaml"
        config_data = read_yaml(str(config_path))
        config = validate_config(config_data)

        assert isinstance(config, Config)
        assert config.system_config.conf_version == "v1.2.1"
        assert config.character_config.character_name == "Test AI"
        assert (
            config.character_config.agent_config.conversation_agent_choice
            == "basic_memory_agent"
        )
