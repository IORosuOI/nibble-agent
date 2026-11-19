"""User profiles: allergens, dislikes and likes. Stored in a JSON file for now."""
import json
from pathlib import Path

from pydantic import BaseModel

PROFILES_FILE = Path(__file__).with_name("profiles.json")


class Profile(BaseModel):
    id: str
    name: str
    allergens: list[str] = []  # EU allergen families, named as in Marius's taxonomy
    dislikes: list[str] = []
    likes: list[str] = []


def load_profile(profile_id: str) -> Profile:
    profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    for p in profiles:
        if p["id"] == profile_id:
            return Profile(**p)
    raise KeyError(f"No profile with id {profile_id!r}")