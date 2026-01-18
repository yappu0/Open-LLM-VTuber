"""
Tests for chat group manager (chat_group.py).
"""

import pytest

from open_llm_vtuber.chat_group import ChatGroupManager, Group


class TestGroup:
    """Tests for Group dataclass."""

    def test_group_creation(self):
        """Test Group creation."""
        group = Group(
            group_id="group_user1",
            owner_uid="user1",
            members={"user1"},
        )
        assert group.group_id == "group_user1"
        assert group.owner_uid == "user1"
        assert "user1" in group.members

    def test_group_members_is_set(self):
        """Test that members is a set."""
        group = Group(
            group_id="group_test",
            owner_uid="owner",
            members={"owner", "member1", "member2"},
        )
        assert isinstance(group.members, set)
        assert len(group.members) == 3


class TestChatGroupManager:
    """Tests for ChatGroupManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ChatGroupManager for each test."""
        return ChatGroupManager()

    def test_init(self, manager):
        """Test ChatGroupManager initialization."""
        assert manager.client_group_map == {}
        assert manager.groups == {}

    def test_create_group_for_client(self, manager):
        """Test creating a group for a client."""
        group_id = manager.create_group_for_client("user1")

        assert group_id == "group_user1"
        assert "user1" in manager.client_group_map
        assert manager.client_group_map["user1"] == "group_user1"
        assert "group_user1" in manager.groups

        group = manager.groups["group_user1"]
        assert group.owner_uid == "user1"
        assert "user1" in group.members

    def test_create_multiple_groups(self, manager):
        """Test creating groups for multiple clients."""
        manager.create_group_for_client("user1")
        manager.create_group_for_client("user2")

        assert len(manager.groups) == 2
        assert "group_user1" in manager.groups
        assert "group_user2" in manager.groups


class TestAddClientToGroup:
    """Tests for add_client_to_group method."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with initial client."""
        mgr = ChatGroupManager()
        # Register clients in the map (simulating connection)
        mgr.client_group_map["user1"] = ""
        mgr.client_group_map["user2"] = ""
        mgr.client_group_map["user3"] = ""
        return mgr

    def test_add_client_creates_group_if_needed(self, manager):
        """Test that adding a client creates a group if inviter has none."""
        success, message = manager.add_client_to_group("user1", "user2")

        assert success is True
        assert "group_user1" in manager.groups
        assert "user1" in manager.groups["group_user1"].members
        assert "user2" in manager.groups["group_user1"].members

    def test_add_client_to_existing_group(self, manager):
        """Test adding a client to an existing group."""
        manager.create_group_for_client("user1")
        success, message = manager.add_client_to_group("user1", "user2")

        assert success is True
        group = manager.groups["group_user1"]
        assert len(group.members) == 2
        assert "user2" in group.members

    def test_add_nonexistent_client(self, manager):
        """Test adding a client that doesn't exist."""
        success, message = manager.add_client_to_group("user1", "nonexistent")

        assert success is False
        assert "does not exist" in message

    def test_add_client_already_in_group(self, manager):
        """Test adding a client who is already in a group."""
        manager.create_group_for_client("user1")
        manager.add_client_to_group("user1", "user2")

        # Try to add user2 to another group
        manager.client_group_map["user3"] = ""
        success, message = manager.add_client_to_group("user3", "user2")

        assert success is False
        assert "already in a group" in message

    def test_add_multiple_clients(self, manager):
        """Test adding multiple clients to a group."""
        manager.add_client_to_group("user1", "user2")
        manager.add_client_to_group("user1", "user3")

        group = manager.groups["group_user1"]
        assert len(group.members) == 3
        assert "user1" in group.members
        assert "user2" in group.members
        assert "user3" in group.members


class TestRemoveClientFromGroup:
    """Tests for remove_client_from_group method."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with a group."""
        mgr = ChatGroupManager()
        mgr.client_group_map["user1"] = ""
        mgr.client_group_map["user2"] = ""
        mgr.client_group_map["user3"] = ""
        mgr.add_client_to_group("user1", "user2")
        mgr.add_client_to_group("user1", "user3")
        return mgr

    def test_owner_removes_member(self, manager):
        """Test owner removing a member."""
        success, message = manager.remove_client_from_group("user1", "user2")

        assert success is True
        group = manager.groups["group_user1"]
        assert "user2" not in group.members
        assert manager.client_group_map["user2"] == ""

    def test_member_removes_self(self, manager):
        """Test member removing themselves."""
        success, message = manager.remove_client_from_group("user2", "user2")

        assert success is True
        assert "user2" not in manager.groups["group_user1"].members

    def test_non_owner_cannot_remove_others(self, manager):
        """Test that non-owner cannot remove others."""
        success, message = manager.remove_client_from_group("user2", "user3")

        assert success is False
        assert "Only group owner or self" in message

    def test_remove_from_nonexistent_group(self, manager):
        """Test removing client not in any group."""
        manager.client_group_map["user4"] = ""
        success, message = manager.remove_client_from_group("user1", "user4")

        assert success is False
        assert "not in any group" in message

    def test_remove_last_member_deletes_group(self, manager):
        """Test that removing all members deletes the group."""
        # Remove all except one
        manager.remove_client_from_group("user1", "user2")
        manager.remove_client_from_group("user1", "user3")
        # Remove last member (owner)
        manager.remove_client_from_group("user1", "user1")

        assert "group_user1" not in manager.groups


class TestRemoveClient:
    """Tests for remove_client method (disconnect handling)."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with a group."""
        mgr = ChatGroupManager()
        mgr.client_group_map["user1"] = ""
        mgr.client_group_map["user2"] = ""
        mgr.client_group_map["user3"] = ""
        mgr.add_client_to_group("user1", "user2")
        mgr.add_client_to_group("user1", "user3")
        return mgr

    def test_remove_member(self, manager):
        """Test removing a regular member."""
        affected = manager.remove_client("user2")

        assert "user2" in affected
        assert "user2" not in manager.client_group_map
        assert "user2" not in manager.groups["group_user1"].members

    def test_remove_owner_reassigns(self, manager):
        """Test that removing owner reassigns ownership."""
        original_owner = manager.groups["group_user1"].owner_uid
        assert original_owner == "user1"

        affected = manager.remove_client("user1")

        assert "user1" in affected
        assert "group_user1" in manager.groups  # Group still exists
        new_owner = manager.groups["group_user1"].owner_uid
        assert new_owner != "user1"
        assert new_owner in ["user2", "user3"]

    def test_remove_last_member_deletes_group(self):
        """Test that removing the last member deletes the group."""
        manager = ChatGroupManager()
        manager.create_group_for_client("user1")

        manager.remove_client("user1")

        assert "group_user1" not in manager.groups

    def test_remove_client_not_in_group(self, manager):
        """Test removing client not in any group."""
        manager.client_group_map["user4"] = ""
        affected = manager.remove_client("user4")

        assert affected == []


class TestGetClientGroup:
    """Tests for get_client_group method."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with a group."""
        mgr = ChatGroupManager()
        mgr.create_group_for_client("user1")
        return mgr

    def test_get_client_group(self, manager):
        """Test getting client's group."""
        group = manager.get_client_group("user1")

        assert group is not None
        assert group.group_id == "group_user1"

    def test_get_client_group_no_group(self, manager):
        """Test getting group for client not in any group."""
        group = manager.get_client_group("user2")

        assert group is None

    def test_get_client_group_empty_string(self, manager):
        """Test getting group when client_group_map has empty string."""
        manager.client_group_map["user2"] = ""
        group = manager.get_client_group("user2")

        assert group is None


class TestGetGroupMembers:
    """Tests for get_group_members method."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with a group."""
        mgr = ChatGroupManager()
        mgr.client_group_map["user1"] = ""
        mgr.client_group_map["user2"] = ""
        mgr.add_client_to_group("user1", "user2")
        return mgr

    def test_get_group_members(self, manager):
        """Test getting group members."""
        members = manager.get_group_members("user1")

        assert len(members) == 2
        assert "user1" in members
        assert "user2" in members

    def test_get_group_members_no_group(self, manager):
        """Test getting members when client not in group."""
        members = manager.get_group_members("user3")

        assert members == []


class TestGetGroupById:
    """Tests for get_group_by_id method."""

    @pytest.fixture
    def manager(self):
        """Create ChatGroupManager with a group."""
        mgr = ChatGroupManager()
        mgr.create_group_for_client("user1")
        return mgr

    def test_get_group_by_id(self, manager):
        """Test getting group by ID."""
        group = manager.get_group_by_id("group_user1")

        assert group is not None
        assert group.owner_uid == "user1"

    def test_get_nonexistent_group(self, manager):
        """Test getting nonexistent group."""
        group = manager.get_group_by_id("nonexistent")

        assert group is None


class TestCleanupDisconnectedClients:
    """Tests for cleanup_disconnected_clients method."""

    def test_cleanup_disconnected(self):
        """Test cleaning up disconnected clients."""
        manager = ChatGroupManager()
        manager.client_group_map["user1"] = ""
        manager.client_group_map["user2"] = ""
        manager.client_group_map["user3"] = ""
        manager.add_client_to_group("user1", "user2")
        manager.add_client_to_group("user1", "user3")

        # Simulate user2 and user3 disconnecting
        connected_clients = {"user1"}
        manager.cleanup_disconnected_clients(connected_clients)

        # user2 and user3 should be removed from group
        group = manager.groups["group_user1"]
        assert "user1" in group.members
        assert "user2" not in manager.client_group_map or manager.client_group_map.get("user2") == ""
        assert "user3" not in manager.client_group_map or manager.client_group_map.get("user3") == ""

    def test_cleanup_all_disconnected_deletes_group(self):
        """Test that group is deleted when all members disconnect."""
        manager = ChatGroupManager()
        manager.create_group_for_client("user1")

        connected_clients = set()  # No one connected
        manager.cleanup_disconnected_clients(connected_clients)

        assert "group_user1" not in manager.groups
