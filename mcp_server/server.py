"""Nibble MCP server. Each tool is a thin wrapper; the logic lives in its owner's folder."""
from mcp.server.mcpserver import MCPServer

from backend.profiles import Profile, load_profile

mcp = MCPServer("nibble")


@mcp.tool()
def get_profile(profile_id: str) -> Profile:
    """Return the user's allergens, dislikes and likes."""
    return load_profile(profile_id)


if __name__ == "__main__":
    mcp.run()