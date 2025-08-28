from typing import Any, Dict, Set

from legoml.core.context import Context
from legoml.core.state import EngineState


def implements(*method_names):
    """Dec
    orator
    to
    mark
    which
    callba
    ck met
    hods a
    class
    implem
    ents.
    """

    def decorator(cls):
        cls._implemented_methods = set(method_names)
        return cls

    return decorator


class Callback:
    """
    Pro
    tocol
    defini
    ng the
    callba
    ck int
    erface
    for
    engine
events.
    All
    method
    s are
    option
    al - c
    allbac
    ks
    only
    need
    to
implement
    the
    events
    they
    care
    about.
    """

    _implemented_methods: Set[str]

    def state_dict(self) -> Dict[str, Any]:
        """R
        et
        ur
        n
        ca
        ll
        ba
        ck
        st
        at
        e
        fo
        r
        ch
        ec
        kp
        oi
        nt
        in
        g.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """L
        oa
        d
        ca
        ll
        ba
        ck
        st
        at
        e
        fr
        om
        ch
        ec
        kp
        oi
        nt
        .
        """
        pass

    def on_engine_start(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        wh
        en
        th
        e
        en
        gi
        ne
        st
        ar
        ts
        tr
        ai
        ni
        ng
        .
        """
        pass

    def on_engine_end(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        wh
        en
        th
        e
        en
        gi
        ne
        fi
        ni
        sh
        es
        tr
        ai
        ni
        ng
        .
        """
        pass

    def on_epoch_start(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        at
        th
        e
        st
        ar
        t
        of
        ea
        ch
        ep
        oc
        h.
        """
        pass

    def on_epoch_end(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        at
        th
        e
        en
        d
        of
        ea
        ch
        ep
        oc
        h.
        """
        pass

    def on_step_start(self, context: Context, state: EngineState, batch: Any) -> None:
        """C
        al
        le
        d
        be
        fo
        re
        pr
        oc
        es
        si
        ng
        ea
        ch
        ba
        tc
        h.
        """
        pass

    def on_step_end(self, context: Context, state: EngineState, batch: Any) -> None:
        """C
        al
        le
        d
        af
        te
        r
        pr
        oc
        es
        si
        ng
        ea
        ch
        ba
        tc
        h.
        St
        ep
        ou
        tp
        ut
        s
available
        in
        st
        at
        e.
        ou
        tp
        ut
        s.
        """
        pass

    def on_backward_start(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        be
        fo
        re
        ba
        ck
        wa
        rd
        pa
        ss
        .
        """
        pass

    def on_backward_end(self, context: Context, state: EngineState) -> None:
        """C
        al
        le
        d
        af
        te
        r
        ba
        ck
        wa
        rd
        pa
        ss
        .
        """
        pass
