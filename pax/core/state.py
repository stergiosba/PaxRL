import equinox as eqx


class EnvState(eqx.Module):
    """The environment state class.

    `Args`:
        - `t (int)`: The current time step.
    """

    time: int

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
