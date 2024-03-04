import optax
import equinox as eqx

class TrainState(eqx.Module):
    params: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    
    def __init__(self, params, optimizer, opt_state = None):

        self.params = params

        self.optimizer = optimizer
        if opt_state is None:
            self.opt_state = self.optimizer.init(self.params)
        else:
            self.opt_state = opt_state

    def apply_gradients(self, grads):

        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        new_params = eqx.apply_updates(self.params, updates)

        new_train_state = self.__class__(params=new_params, optimizer=self.optimizer, opt_state=opt_state)
        return new_train_state
