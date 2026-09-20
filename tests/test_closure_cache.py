"""The transitive closure is rebuilt for every processor, which dominates the run.

Caching it is only safe while nothing that feeds it has changed: the definitions
themselves and the name pointers between them.
"""

from pycg_ml.machinery.definitions import DefinitionManager
from pycg_ml.utils import constants


def manager_with_chain():
    manager = DefinitionManager()
    manager.create("main.a", constants.NAME_DEF)
    manager.create("main.b", constants.NAME_DEF)
    manager.get("main.a").get_name_pointer().add("main.b")
    return manager


def test_closure_is_reused_when_nothing_changed():
    manager = manager_with_chain()

    first = manager.transitive_closure()
    second = manager.transitive_closure()

    assert first is second


def test_a_new_pointer_invalidates_the_closure():
    manager = manager_with_chain()
    manager.create("main.c", constants.NAME_DEF)
    before = manager.transitive_closure()

    manager.get("main.b").get_name_pointer().add("main.c")
    after = manager.transitive_closure()

    assert after is not before
    assert after["main.a"] == {"main.c"}


def test_a_new_definition_invalidates_the_closure():
    manager = manager_with_chain()
    before = manager.transitive_closure()

    manager.create("main.d", constants.NAME_DEF)
    after = manager.transitive_closure()

    assert after is not before
    assert "main.d" in after


def test_merging_a_definition_invalidates_the_closure():
    manager = manager_with_chain()
    manager.create("main.e", constants.NAME_DEF)
    manager.get("main.e").get_name_pointer().add("main.b")
    before = manager.transitive_closure()

    manager.assign("main.f", manager.get("main.e"))
    after = manager.transitive_closure()

    assert after is not before
    assert after["main.f"] == {"main.b"}
