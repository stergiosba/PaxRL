import optax
import equinox as eqx

class TrainState(eqx.Module):
    model: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    
    def __init__(self, model, optimizer, opt_state = None):
        self.model = model
        self.optimizer = optimizer
        if opt_state is None:
            self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        else:
            self.opt_state = opt_state

    def apply_gradients(self, grads):

        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.model)
        model = eqx.apply_updates(self.model, updates)
        new_train_state = self.__class__(model=model, optimizer=self.optimizer, opt_state=opt_state)
        return new_train_state