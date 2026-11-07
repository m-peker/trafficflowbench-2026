from queue_ongoing import fit_model
from config import OUT
model, settings = fit_model()
model.save_model(str(OUT/'ongoing_lgb.txt'))
