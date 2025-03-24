import nedc_ml_tools as mlt
import nedc_ml_tools_data as mltd

params = [{
    'npts': 100,
    'mean': [0, 0],
    'cov': [[1, 0.5], [0.5, 1]]
}]
data = mltd.MLToolsData.generate_data('gaussian', params)

model = mlt.Alg()

model.set('TRANSFORMER')

model.set_parameters(
    {"implementation": "pytorch",
     "epoch": 5,
     "learning_rate": 0.001,
     "batch_size": 32,
     "embed_size": 32,
     "nheads": 2,
     "num_layers": 2,
     "mlp_dim": 4,
     "dropout": 0.1,
     "random_state": 42})

model.train(data)

labels, probs = model.predict(data)

print(labels)