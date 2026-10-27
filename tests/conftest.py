import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oee import llm
from oee.config import Settings
from oee.engine import Engine
from tests.stub_provider import StubProvider

# the in-test provider double: unit tests never touch a network endpoint, and
# the product itself ships no offline stand-in
llm.PROVIDER_REGISTRY.setdefault("stub", StubProvider)


def stub_settings(db_path: str) -> Settings:
    s = Settings()
    s.provider = "stub"
    s.db_path = db_path
    s.embed_provider = "local"
    return s



@pytest.fixture
def engine(tmp_path):
    eng = Engine(stub_settings(str(tmp_path / "test.db")))
    yield eng
    eng.store.close()


DOC = (
    "Page 1. The Kessel circuit self-calibrates its threshold voltage using a "
    "single polysilicon sense line. The group measured drift under bias stress at "
    "85C and reported a 40 percent improvement over the uncalibrated control. "
    "No independent replication is recorded here, and the measurement window was "
    "200 hours. Therefore the authors claim the drift failure mode is removed.\n\n"
    "Page 2. Deployment notes. Reader panels failed at 11 percent per year, and the "
    "failures traced to moisture ingress in the encapsulant, not threshold drift. "
    "The logbook assumes the 1998 study measured the operative failure mode; it did not. "
    "Line engineers recorded lot numbers and bake schedules for six months.\n\n"
    "Page 3. Administrative appendix. The budget committee reviewed procurement, "
    "travel, and the monthly scrap rate. Nothing in the appendix measures threshold "
    "drift, and no measurement of moisture ingress is reported."
)
