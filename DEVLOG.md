# DEVLOG

## Week 1
- Did: Created repo, wrote project plan and 3 proposals.
- Result: Repo is public, first commit done.
- Next: Get topic approved, load data in W2.
- Blocked: Nothing yet.


## Week 2
- Did: Got topic approved. Wrote DATA.md. Wrote load_data.py. Proved 1,200 rows load.
- Result: Checkpoint F passed — data loads.
- Next: Build a dumb baseline in W3.
- Blocked: Nothing.

- 
## Week 3
- Did: Built the data pipeline. Created generate_data.py, src/data/dataset.py, src/train.py, and data_card.md. Ran the training script successfully.
- Result: Data loads, cleans, and feeds into a PyTorch DataLoader. One epoch runs without errors.
- Next: Build the baseline model in Week 4.
- Blocked: Nothing.

- ## Week 4
- Did: Built an MLP model with nn.Module and ran an ablation study on make_moons/make_circles datasets. Integrated the model into src/train.py.
- Result: Model v0 is working. Ablation table shows that ReLU activation is crucial for the circles dataset.
- Next: Start the capstone project's model development in Week 5.
- Blocked: Nothing.
