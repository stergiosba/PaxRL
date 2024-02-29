import optax
import equinox as eqx

class TrainState(eqx.Module):
    params: eqx.Module
    static: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    
    def __init__(self, model, optimizer, opt_state = None):

        self.params, self.static  = eqx.partition(model, eqx.is_array)

        self.optimizer = optimizer
        if opt_state is None:
            self.opt_state = self.optimizer.init(self.params)
        else:
            self.opt_state = opt_state

    def apply_gradients(self, grads):

        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        params = eqx.apply_updates(self.params, updates)

        new_model = eqx.combine(params, self.static)
        new_train_state = self.__class__(model=new_model, optimizer=self.optimizer, opt_state=opt_state)
        return new_train_state
    
    @property
    def model(self):
        return eqx.combine(self.params, self.static)