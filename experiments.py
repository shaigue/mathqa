"""Running different experiments"""
from train_mathqa import get_manager, get_model, train
import config


def many_macros_experiment():
    n_macros = [21, 41, 61, 81, 100]
    for n in n_macros:
        manager = get_manager(macro_file=config.get_macro_file(n))
        model = get_model(manager)
        name = f'many_macros_{n}'
        dir_path = config.get_exp_dir_path(name)
        train(
            dir_path=dir_path,
            model=model,
            manager=manager,
            n_epochs=250,
            evaluate_every=10,
        )
