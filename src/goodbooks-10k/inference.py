from model.collaborative import als_model
from utils.evaluate import Evaluate


total_rec_list, gt = als_model()
evaluate_func = Evaluate(total_rec_list, gt, topn=200)
evaluate_func._evaluate()
