from app.training.reproducibility import seed_everything

def test_seed_everything_returns_seed(): assert seed_everything(7)['seed'] == 7
