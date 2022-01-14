# svae_spectra
Source code for the paper "Bi-modal Variational Autoencoders for Metabolite Identification Using Tandem Mass Spectrometry"

Source code for training and evaluating SVAE and JMVAE models is provided for spectra-SMILES and spectra-fingerprints case.
`data` folder contains the list of spectra identifiers used for training and testing and the example data for demonstrating the required format for testing and training.
`experiments` folder contains the code for spectra preprocessing, model training and testing.
`models` folder contains the bimodal VAE model classes.

`data/train_ids.txt` is a list of MoNA and NIST identifiers used for training in the paper evaluations.
`data/test_ids.txt` is a list of MoNA and NIST identifiers used for testing in the paper evaluations.

```
usage: run.py [-h] -t TRAIN -v TEST [-u TRAIN_MOLECULES] -m MODEL [-k KEYWORD]
              [-l LOAD] [-d DEVICE] [-e]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Train set, a csv file with comma as delimiter,
                        required columns 'spectrum' and 'SMILES'
  -v TEST, --test TEST  Test set, a csv file with comma as delimiter, required
                        columns 'spectrum' and 'SMILES'
  -u TRAIN_MOLECULES, --train_molecules TRAIN_MOLECULES
                        Unsupervised molecule train set. If not provided, the
                        algorithm runs in supervised mode
  -m MODEL, --model MODEL
                        Model name (SVAE or JMVAE). JMVAE only runs in
                        supervised mode even if train_molecules provided
  -k KEYWORD, --keyword KEYWORD
                        Additional keyword for the trained model name. Default
                        'molecules'
  -l LOAD, --load LOAD  Trained model name to evaluate or continue the
                        training
  -d DEVICE, --device DEVICE
                        Device name, default cpu
  -e, --eval            If eval flag is set, model predicts spectra from
                        molecules for 1000 samples from the test set and
                        predicts molecules from spectra
```

Example of training the JMVAE model for fingerprints in supervised mode:
```
python experiments/spectra_fingerprints_paper/run.py --train data/train.csv --test data/test.csv --model JMVAE --device cuda:0
```

Example of training the SVAE model for SMILES in semi-supervised mode:
```
python experiments/spectra/run.py --train data/train.csv --test data/test.csv --train_molecules data/unsupervised.csv --model SVAE --device cuda:0
```

Example of evaluating the trained SVAE model for SMILES:
```
python experiments/spectra/run.py --train train.csv --test data/test.csv --load trained_model.pth.tar --model SVAE --device cuda:0 --eval
```