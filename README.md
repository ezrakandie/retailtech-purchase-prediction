# RetailTech Purchase Prediction

A PyTorch neural network that predicts whether an e-commerce customer 
will complete a purchase, based on browsing session behavior (time on 
site, pages viewed, basket value, device type, customer type).

## Pipeline
1. **Data cleaning** — handled missing values with column-appropriate 
   strategies (median for time spent, mean for pages viewed, 0 for 
   basket value, categorical fill for device/customer type).
2. **Feature engineering** — min-max scaled numerical features, 
   one-hot encoded categorical features.
3. **Model** — a small feedforward network (1 hidden layer, 8 units, 
   ReLU activation, sigmoid output) trained with binary cross-entropy 
   loss using the Adam optimizer.

## Results
Held-out validation accuracy: ~78.75%

## What I'd improve
- Compare against a simpler baseline (e.g. logistic regression) to 
  check whether the neural net is actually earning its added complexity
- Check class balance on the `purchase` target — if it's skewed, 
  accuracy alone is a misleading metric
- Track precision/recall in addition to accuracy, since false 
  negatives (missing a likely buyer) and false positives probably 
  carry different costs for a marketing team
- Use k-fold cross-validation instead of a single train/test split for 
  a more robust accuracy estimate

## Setup
```bash
pip install -r requirements.txt
```
Then open `notebook.ipynb` and run the cells in order.

## Data
`data/*.csv` files are synthetic data created for a training exercise — 
not real customer records.
