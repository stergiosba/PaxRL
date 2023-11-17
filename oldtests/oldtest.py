import time
import jax
import jax.random as jrandom
import jax.numpy as jnp
import jax.numpy.linalg as la
import numpy as np
import pax.training as tpax
import matplotlib
import matplotlib.pyplot as plt
from pax.training.models import Agent
from pax import make
from pax.render.render import render, matplotlib_render
matplotlib.use('Qt5Agg')


def test_1(console_args):
    env_name = "prob_env"
    env = make(env_name)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    trainer = tpax.Trainer(env)

    s = time.time()
    if console_args.profile == "y":
        with jax.profiler.trace("/tmp/tensorboard"):
            # Run the operations to be profiled
            batch = trainer(model, key, "train_cfg")
            #(obs, state, act, batch_reward) = trainer(key, "train_cfg")
    else:
        num_total_epochs, log_step, log_returns = trainer(model, key, "train_cfg")

    fig, ax = plt.subplots()
    t = np.arange(0.0, len(log_returns), 1.0)
    ax.plot(t, log_returns)
    ax.grid()
    plt.show()
    # for en in range(env.params.settings["n_env"]):
    #     t = np.arange(0.0, env.params.scenario["episode_size"], 1.0)
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(0, 800)  # Set x-axis limits
    #     ax.set_ylim(0, 800)
    #     for ag in range(env.params.scenario["n_scripted_entities"]):
    #         if ag == 13:
    #             ax.plot(obs[:, en, ag, 0], obs[:, en, ag, 1], marker= "o")
    #         else:
    #             #pass
    #             ax.plot(obs[:, en, ag, 0], obs[:, en, ag, 1])
    #     ax.grid()
    #     mats=plt.matshow(np.diag(np.linalg.norm(obs[999,en,:-1,:]-obs[999,en,-1,:],axis=-1)))
    #     fig.savefig(f"{en}_leader_all.png")
    # print(state.X_dot[20,2,1,:])
    # print(state.X_dot_prev[20,2,1,:])
    # v1 = la.norm(state.X_dot[20,2,1,:])
    # v2 = la.norm(state.X_dot_prev[20,2,1,:])
    # print(jnp.arccos(state.X_dot[20,2,1,:].T@state.X_dot_prev[20,2,1,:]/(v1*v2))*180/jnp.pi)
    # print(state.B>1)
    # plt.show()
    print(f"Time for trainer: {time.time()-s}")


    if console_args.render in ["human", "Human", "h", "H"]:
        render(env, state, record=env.params.settings["record"])

    elif console_args.render in ["matplotlib", "m", "M", "mat"]:
        matplotlib_render(env, state)


TESTS = {"1": test_1}

def run(console_args):
    s = time.time()
    #TESTS[console_args.test](console_args)
    TESTS["1"](console_args)
    print(f"Total time:{time.time()-s}")
