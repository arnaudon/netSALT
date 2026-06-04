"""Pydantic model for the ``graph.graph["params"]`` configuration.

Historically ``params`` was a bare dict passed around the codebase. That made
typos silent and offered no validation at the seam where users hand-write
config values. The :class:`NetSaltParams` model below replaces that
dict while keeping full dict-compatible access (``params["k_min"]``,
``params.get(...)``, ``"x" in params``) so every existing call site continues
to work. Validation runs at construction time (via
:meth:`NetSaltParams.from_dict`) and on assignment
(``params["k_min"] = 1.5`` goes through the field validator).
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class NetSaltParams(BaseModel):
    """Typed configuration for a quantum graph.

    Unknown keys are allowed (``extra="allow"``) so users can stash
    problem-specific knobs without editing this model; known keys get type
    validation. All fields default to ``None`` — the codebase already uses
    ``params.get(...)`` with per-call defaults, so a required-fields model
    would be too strict without a wider refactor.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # --- Complex-frequency scan grid ---------------------------------------
    k_min: float | None = None
    k_max: float | None = None
    k_n: int | None = None
    alpha_min: float | None = None
    alpha_max: float | None = None
    alpha_n: int | None = None

    # --- Lasing / gain physics ---------------------------------------------
    k_a: float | None = None
    gamma_perp: float | None = None
    D0: float | None = None
    D0_max: float | None = None
    D0_steps: int | None = None

    # --- Dispersion / dielectric -------------------------------------------
    # c and dielectric_constant can be scalars or arrays; typed as Any to
    # avoid forcing a conversion on assignment.
    c: Any | None = None
    dielectric_params: dict | None = None
    refraction_params: dict | None = None
    dielectric_constant: Any | None = None

    # --- Graph structure / pump --------------------------------------------
    open_model: str | None = None
    inner: list | None = None
    pump: Any | None = None

    # --- Mode-refinement knobs ---------------------------------------------
    # ``mode_search_method`` picks how :func:`netsalt.find_passive_modes`
    # locates modes in the scan rectangle. Accepted values:
    #   ``"contour"`` (default) — Beyn's contour integration via
    #     ``find_modes_contour``. Directly returns modes at
    #     ``|λ₁| ≲ 1e-8`` with no per-mode refinement step.
    #   ``"grid"`` — legacy path: ``scan_frequencies`` + ``find_modes``
    #     (grid scan + ``peak_local_max`` + ``refine_mode``). Kept for
    #     backward compatibility and for callers who want to visualise
    #     the quality field.
    mode_search_method: Literal["contour", "grid"] | None = None
    # ``refine_method`` picks the algorithm used by :func:`netsalt.refine_mode`
    # when refinement is explicitly invoked — primarily by
    # ``pump_trajectories`` and ``find_threshold_lasing_modes`` tracking a
    # single mode as ``D0`` varies, not for the passive-scan step.
    # Typed as a Literal so typos in the config fail at the graph
    # boundary (``update_parameters``) rather than silently making it all
    # the way to ``refine_mode`` before raising.
    refine_method: Literal["root", "brownian"] | None = None
    search_stepsize: float | None = None
    quality_threshold: float | None = None
    max_steps: int | None = None
    # Legacy knobs, only read by refine_mode_brownian_ratchet:
    max_tries_reduction: int | None = None
    reduction_factor: float | None = None
    n_modes_max: int | None = None

    # --- Infrastructure ----------------------------------------------------
    n_workers: int | None = None

    # --- Plotting ----------------------------------------------------------
    plot_edgesize: float | None = None
    exts: list | None = None

    # ------------------------------------------------------------------ dict API
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            extra = self.__pydantic_extra__ or {}
            if key in extra:
                return extra[key]
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        # Going through setattr triggers field validation when the key is a
        # declared field; for extras pydantic stores into __pydantic_extra__.
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        if key in type(self).model_fields:
            setattr(self, key, None)
            return
        extra = self.__pydantic_extra__ or {}
        if key in extra:
            del extra[key]
            return
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if key in type(self).model_fields and getattr(self, key) is not None:
            return True
        extra = self.__pydantic_extra__ or {}
        return key in extra

    def __iter__(self):
        yield from self._live_keys()

    def __len__(self) -> int:
        return sum(1 for _ in self._live_keys())

    def _live_keys(self):
        for k in type(self).model_fields:
            if getattr(self, k) is not None:
                yield k
        extra = self.__pydantic_extra__ or {}
        yield from extra

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self[key]
        except KeyError:
            return default
        # Match dict.get semantics: missing means default, but an explicit
        # None stored for a field also behaves as "absent" from the caller's
        # point of view (since every declared field defaults to None).
        return default if value is None else value

    def keys(self):
        return list(self._live_keys())

    def values(self):
        return [self[k] for k in self._live_keys()]

    def items(self):
        return [(k, self[k]) for k in self._live_keys()]

    def update(self, other: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        if other is not None:
            for k, v in other.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def pop(self, key: str, *default: Any) -> Any:
        try:
            value = self[key]
        except KeyError:
            if default:
                return default[0]
            raise
        del self[key]
        return value

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: "NetSaltParams | Mapping[str, Any] | None") -> "NetSaltParams":
        """Coerce a dict (or existing model) into a validated NetSaltParams.

        Accepts None as an empty dict — matches the old behaviour of callers
        that passed ``params=None`` to :func:`create_quantum_graph`.
        """
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls.model_validate(dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy suitable for JSON / YAML serialisation."""
        return {k: self[k] for k in self._live_keys()}
