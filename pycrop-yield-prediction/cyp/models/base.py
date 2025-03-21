import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime
from ..data.utils import convert_kg_per_ha_to_bu_per_acre, standardize_yield_data
from .gp import GaussianProcess
from .loss import huber_loss #l1_l2_loss


# Add to the existing ModelBase.__init__ method:




class ModelBase:
    """
    Base class for all models
    """
    # def __init__(self, model, model_weight, model_bias, model_type, savedir, use_gp=True,
    #              sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
    #              #device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #              device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")):
    def __init__(self, model, model_weight, model_bias, model_type, savedir, use_gp=True,
             sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
             use_sparse_gp=False, num_inducing=500, sparse_method='fitc',
             #device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f'Using {device.type}')
        if device.type != 'cpu':
            #model = model.cuda()
            model = model.to(self.device)
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        self.min_val = None
        self.max_val = None
        
        # for reproducability
        torch.manual_seed(42)
        #torch.cuda.manual_seed_all(42)

        # self.gp = None
        # if use_gp:
        #     self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)
        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b,
                                    use_sparse=use_sparse_gp,
                                    num_inducing=num_inducing,
                                    sparse_method=sparse_method)

    # def run(self, path_to_histogram=Path('data/img_output/histogram_all_full.npz'),
    #         times='all', pred_years=None, num_runs=2, train_steps=25000, batch_size=32,
    #         starter_learning_rate=1e-3, weight_decay=0, l1_weight=0, patience=10):
    #     """
    #     Train the models. Note that multiple models are trained: as per the paper, a model
    #     is trained for each year, with all preceding years used as training values. In addition,
    #     for each year, 2 models are trained to account for random initialization.

    #     Parameters
    #     ----------
    #     path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
    #         The location of the training data
    #     times: {'all', 'realtime'}
    #         Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
    #         If 'realtime', range(10, 31, 4) is used.
    #     pred_years: int, list or None, default=None
    #         Which years to build models for. If None, the default values from the paper (range(2009, 2016))
    #         are used.
    #     num_runs: int, default=2
    #         The number of runs to do per year. Default taken from the paper
    #     train_steps: int, default=25000
    #         The number of steps for which to train the model. Default taken from the paper.
    #     batch_size: int, default=32
    #         Batch size when training. Default taken from the paper
    #     starter_learning_rate: float, default=1e-3
    #         Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
    #         steps. Default taken from the paper
    #     weight_decay: float, default=1
    #         Weight decay (L2 regularization) on the model weights
    #     l1_weight: float, default=0
    #         In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
    #     patience: int or None, default=10
    #         The number of epochs to wait without improvement in the validation loss before terminating training.
    #         Note that the original repository doesn't use early stopping.
    #     """

    #     with np.load(path_to_histogram) as hist:
    #         images = hist['output_image']
    #         locations = hist['output_locations']
    #         yields = hist['output_yield']
    #         years = hist['output_year']
    #         indices = hist['output_index']

    #     # to collect results
    #     years_list, run_numbers, rmse_list, me_list, times_list = [], [], [], [], []
    #     if self.gp is not None:
    #         rmse_gp_list, me_gp_list = [], []

    #     if pred_years is None:
    #         pred_years = range(2009, 2018)
    #     elif type(pred_years) is int:
    #         pred_years = [pred_years]

    #     if times == 'all':
    #         times = [32]
    #     else:
    #         times = range(10, 31, 4)

    #     for pred_year in pred_years:
    #         for run_number in range(1, num_runs + 1):
    #             for time in times:
    #                 print(f'Training to predict on {pred_year}, Run number {run_number}')

    #                 results = self._run_1_year(images, yields,
    #                                            years, locations,
    #                                            indices, pred_year,
    #                                            time, run_number,
    #                                            train_steps, batch_size,
    #                                            starter_learning_rate,
    #                                            weight_decay, l1_weight,
    #                                            patience)

    #                 years_list.append(pred_year)
    #                 run_numbers.append(run_number)
    #                 times_list.append(time)

    #                 if self.gp is not None:
    #                     rmse, me, rmse_gp, me_gp = results
    #                     rmse_gp_list.append(rmse_gp)
    #                     me_gp_list.append(me_gp)
    #                 else:
    #                     rmse, me = results
    #                 rmse_list.append(rmse)
    #                 me_list.append(me)
    #             print('-----------')

    #     # save results to a csv file
    #     data = {'year': years_list, 'run_number': run_numbers, 'time_idx': times_list,
    #             'RMSE': rmse_list, 'ME': me_list}
    #     if self.gp is not None:
    #         data['RMSE_GP'] = rmse_gp_list
    #         data['ME_GP'] = me_gp_list
    #     results_df = pd.DataFrame(data=data)
    #     results_df.to_csv(self.savedir / f'{str(datetime.now())}.csv', index=False)

    def run(self, path_to_histogram=Path('data/img_output/histogram_all_full.npz'),
            times='all', pred_years=None, num_runs=2, train_steps=25000, batch_size=32,
            starter_learning_rate=1e-3, weight_decay=0, l1_weight=0, patience=10):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.

        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int, list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.
        """

        with np.load(path_to_histogram) as hist:
            images = hist['output_image']
            locations = hist['output_locations']
            yields = hist['output_yield']
            years = hist['output_year']
            indices = hist['output_index']

        # to collect results
        years_list, run_numbers, rmse_list, me_list, times_list = [], [], [], [], []
        if self.gp is not None:
            rmse_gp_list, me_gp_list = [], []

        if pred_years is None:
            pred_years = range(2009, 2018)
        elif type(pred_years) is int:
            pred_years = [pred_years]

        if times == 'all':
            times = [32]
        else:
            times = range(10, 31, 4)

        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                for time in times:
                    print(f'Training to predict on {pred_year}, Run number {run_number}')

                    results = self._run_1_year(images, yields,
                                               years, locations,
                                               indices, pred_year,
                                               time, run_number,
                                               train_steps, batch_size,
                                               starter_learning_rate,
                                               weight_decay, l1_weight,
                                               patience)

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(time)

                    if self.gp is not None:
                        rmse, me, r2, rmse_t_ha, rmse_gp, me_gp, r2_gp, rmse_gp_t_ha = results
                        rmse_gp_list.append(rmse_gp)
                        me_gp_list.append(me_gp)
                    else:
                        rmse, me, r2, rmse_t_ha = results
                    rmse_list.append(rmse)
                    me_list.append(me)
                print('-----------')

        # save results to a csv file
        data = {'year': years_list, 'run_number': run_numbers, 'time_idx': times_list,
                'RMSE': rmse_list, 'ME': me_list}
        if self.gp is not None:
            data['RMSE_GP'] = rmse_gp_list
            data['ME_GP'] = me_gp_list
        results_df = pd.DataFrame(data=data)
        results_df.to_csv(self.savedir / f'{str(datetime.now())}.csv', index=False)

    def _run_1_year(self, images, yields, years, locations, indices, predict_year, time, run_number,
                    train_steps, batch_size, starter_learning_rate, weight_decay, l1_weight, patience):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """
        train_data, test_data = self.prepare_arrays(images, yields, locations,
                                                    indices, years, predict_year, time)

        # reinitialize the model, since self.model may be trained multiple
        # times in one call to run()
        self.reinitialize_model(time=time)

        train_scores, val_scores = self._train(train_data.images, train_data.yields,
                                               train_steps, batch_size,
                                               starter_learning_rate,
                                               weight_decay, l1_weight,
                                               patience)

        results = self._predict(*train_data, *test_data, batch_size)

        model_information = {
            'state_dict': self.model.state_dict(),
            'val_loss': val_scores['loss'],
            'train_loss': train_scores['loss'],
        }
        for key in results:
            model_information[key] = results[key]

        # finally, get the relevant weights for the Gaussian Process
        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != 'cpu':
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information['model_weight'] = model_weight.numpy()
        model_information['model_bias'] = model_bias.numpy()

        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(model_information['train_feat'],
                                  model_information['test_feat'],
                                  model_information['train_loc'],
                                  model_information['test_loc'],
                                  model_information['train_years'],
                                  model_information['test_years'],
                                  model_information['train_real'],
                                  model_information['model_weight'],
                                  model_information['model_bias'])
            model_information['test_pred_gp'] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(model_information['test_real'], model_information['test_pred'],
                                    model_information['test_pred_gp'] if self.gp is not None else None)

    def _train(self, train_images, train_yields, train_steps,
           batch_size, starter_learning_rate, weight_decay, l1_weight, patience):
        """Defines the training loop for a model"""

        # split the training dataset into a training and validation set
        total_size = train_images.shape[0]
        print(f'Total size of training set: {total_size}')
        val_size = total_size // 10
        train_size = total_size - val_size
        print(f'After split, training on {train_size} examples, '
            f'validating on {val_size} examples')
        train_dataset, val_dataset = random_split(TensorDataset(train_images, train_yields),
                                                (train_size, val_size))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=starter_learning_rate, weight_decay=weight_decay, eps=1e-8)

        num_epochs = int(train_steps / (train_images.shape[0] / batch_size))
        print(f'Training for {num_epochs} epochs')

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()

        if patience is not None:
            epochs_without_improvement = 0

        for _ in range(num_epochs):
            self.model.train()

            running_train_scores = defaultdict(list)
            
            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                #loss, running_train_scores = l1_l2_loss(pred_y, train_y, l1_weight, running_train_scores)
                loss, running_train_scores = huber_loss(pred_y, train_y, delta=1.0, running_scores=running_train_scores)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)

                optimizer.step()

                train_scores['loss'].append(loss.item())

                step_number += 1

                if step_number in [2000, 5000, 7000, 10000]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for val_x, val_y in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)

                    #val_loss, running_val_scores = l1_l2_loss(val_pred_y, val_y, l1_weight, running_val_scores)
                    val_loss, running_val_scores = huber_loss(val_pred_y, val_y, delta=1.0, running_scores=running_val_scores)

                    val_scores['loss'].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append('{}: {}'.format(key, round(np.array(val).mean(), 5)))

            print('TRAINING: {}'.format(', '.join(train_output_strings)))
            print('VALIDATION: {}'.format(', '.join(val_output_strings)))

            epoch_val_loss = np.array(running_val_scores['loss']).mean()

            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    self.model.load_state_dict(best_state)
                    print('Early stopping!')
                    break
                
            # Step the scheduler at the end of each epoch
            scheduler.step()

        self.model.load_state_dict(best_state)
        return train_scores, val_scores
    
    # def _predict(self, train_images, train_yields, train_locations, train_indices,
    #              train_years, train_additional_feature, test_images, test_yields, test_locations, test_indices,
    #              test_years, test_additional_feature, batch_size):
    #     """
    #     Predict on the training and validation data. Optionally, return the last
    #     feature vector of the model.
    #     """
    #     train_dataset = TensorDataset(train_images, train_yields,
    #                                   train_locations, train_indices,
    #                                   train_years, train_additional_feature)

    #     test_dataset = TensorDataset(test_images, test_yields,
    #                                  test_locations, test_indices,
    #                                  test_years, test_additional_feature)

    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    #     results = defaultdict(list)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for train_im, train_yield, train_loc, train_idx, train_year, train_add_feat in tqdm(train_dataloader):
    #             model_output = self.model(train_im,
    #                                       return_last_dense=True if (self.gp is not None) else False)
    #             if self.gp is not None:
    #                 pred, feat = model_output
    #                 if feat.device != 'cpu':
    #                     feat = feat.cpu()
    #                 results['train_feat'].append(feat.numpy())
    #             else:
    #                 pred = model_output
    #             results['train_pred'].extend(pred.squeeze(1).tolist())
    #             results['train_real'].extend(train_yield.squeeze(1).tolist())
    #             results['train_loc'].append(train_loc.cpu().numpy())
    #             results['train_indices'].append(train_idx.cpu().numpy())
    #             results['train_years'].extend(train_year.tolist())

    #         for test_im, test_yield, test_loc, test_idx, test_year, test_add_feat in tqdm(test_dataloader):
    #             model_output = self.model(test_im,
    #                                       return_last_dense=True if (self.gp is not None) else False)
    #             if self.gp is not None:
    #                 pred, feat = model_output
    #                 if feat.device != 'cpu':
    #                     feat = feat.cpu()
    #                 results['test_feat'].append(feat.numpy())
    #             else:
    #                 pred = model_output
    #             results['test_pred'].extend(pred.squeeze(1).tolist())
    #             results['test_real'].extend(test_yield.squeeze(1).tolist())
    #             results['test_loc'].append(test_loc.cpu().numpy())
    #             results['test_indices'].append(test_idx.cpu().numpy())
    #             results['test_years'].extend(test_year.tolist())

    #     for key in results:
    #         if key in ['train_feat', 'test_feat', 'train_loc',
    #                    'test_loc', 'train_indices', 'test_indices']:
    #             results[key] = np.concatenate(results[key], axis=0)
    #         else:
    #             results[key] = np.array(results[key])
    #     return results
    
    def _predict(self, train_images, train_yields, train_locations, train_indices,
             train_years, test_images, test_yields, test_locations, test_indices,
             test_years, batch_size):
        train_dataset = TensorDataset(train_images, train_yields, train_locations, train_indices, train_years)
        test_dataset = TensorDataset(test_images, test_yields, test_locations, test_indices, test_years)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(train_dataloader):
                model_output = self.model(train_im, return_last_dense=True if (self.gp is not None) else False)
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != 'cpu':
                        feat = feat.cpu()
                    results['train_feat'].append(feat.numpy())
                else:
                    pred = model_output
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    raise ValueError("Model output contains NaN or infinite values")
                results['train_pred'].extend(pred.squeeze(1).tolist())
                results['train_real'].extend(train_yield.squeeze(1).tolist())
                results['train_loc'].append(train_loc.cpu().numpy())
                results['train_indices'].append(train_idx.cpu().numpy())
                results['train_years'].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(test_dataloader):
                model_output = self.model(test_im, return_last_dense=True if (self.gp is not None) else False)
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != 'cpu':
                        feat = feat.cpu()
                    results['test_feat'].append(feat.numpy())
                else:
                    pred = model_output
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    raise ValueError("Model output contains NaN or infinite values")
                results['test_pred'].extend(pred.squeeze(1).tolist())
                results['test_real'].extend(test_yield.squeeze(1).tolist())
                results['test_loc'].append(test_loc.cpu().numpy())
                results['test_indices'].append(test_idx.cpu().numpy())
                results['test_years'].extend(test_year.tolist())

        for key in results:
            if key in ['train_feat', 'test_feat', 'train_loc', 'test_loc', 'train_indices', 'test_indices']:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    # def prepare_arrays(self, images, yields, locations, indices, years, predict_year,
    #                    time):
    #     """Prepares the inputs for the model, in the following way:
    #     - normalizes the images
    #     - splits into a train and val set
    #     - turns the numpy arrays into tensors
    #     - removes excess months, if monthly predictions are being made
    #     """
    #     train_idx = np.nonzero(years < predict_year)[0]
    #     test_idx = np.nonzero(years == predict_year)[0]

    #     train_images, test_images = self._normalize(images[train_idx], images[test_idx])

    #     print(f'Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}')

    #     Data = namedtuple('Data', ['images', 'yields', 'locations', 'indices', 'years'])

    #     train_data = Data(
    #         images=torch.as_tensor(train_images[:, :, :time, :], device=self.device).float(),
    #         yields=torch.as_tensor(yields[train_idx], device=self.device).float().unsqueeze(1),
    #         locations=torch.as_tensor(locations[train_idx]),
    #         indices=torch.as_tensor(indices[train_idx]),
    #         years=torch.as_tensor(years[train_idx])
    #     )

    #     test_data = Data(
    #         images=torch.as_tensor(test_images[:, :, :time, :], device=self.device).float(),
    #         yields=torch.as_tensor(yields[test_idx], device=self.device).float().unsqueeze(1),
    #         locations=torch.as_tensor(locations[test_idx]),
    #         indices=torch.as_tensor(indices[test_idx]),
    #         years=torch.as_tensor(years[test_idx])
    #     )

    #     return train_data, test_data
    
    def prepare_arrays(self, images, yields, locations, indices, years, predict_year, time):
        train_idx = np.nonzero(years < predict_year)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])

        print(f'Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}')

        # Convert yield data from kg/ha to bushels per acre
        yields = convert_kg_per_ha_to_bu_per_acre(yields)

        # Normalize yield data
        #yields, self.yield_min, self.yield_max = normalize_yield_data(yields)
        yields, self.yield_mean, self.yield_std = standardize_yield_data(yields)

        train_images = train_images.reshape(train_images.shape[0], -1, train_images.shape[1])
        test_images = test_images.reshape(test_images.shape[0], -1, test_images.shape[1])

        Data = namedtuple('Data', ['images', 'yields', 'locations', 'indices', 'years'])

        train_data = Data(
            #images=torch.as_tensor(train_images[:, :, :time, :], dtype=torch.float32).to(self.device),
            images=torch.as_tensor(train_images, dtype=torch.float32).to(self.device),
            yields=torch.as_tensor(yields[train_idx], dtype=torch.float32).unsqueeze(1).to(self.device),
            locations=torch.as_tensor(locations[train_idx], dtype=torch.float32).to(self.device),
            indices=torch.as_tensor(indices[train_idx], dtype=torch.float32).to(self.device),
            years=torch.as_tensor(years[train_idx], dtype=torch.float32).to(self.device)
        )

        test_data = Data(
            #images=torch.as_tensor(train_images[:, :, :time, :], dtype=torch.float32).to(self.device),
            images=torch.as_tensor(test_images, dtype=torch.float32).to(self.device),
            yields=torch.as_tensor(yields[test_idx], dtype=torch.float32).unsqueeze(1).to(self.device),
            locations=torch.as_tensor(locations[test_idx], dtype=torch.float32).to(self.device),
            indices=torch.as_tensor(indices[test_idx], dtype=torch.float32).to(self.device),
            years=torch.as_tensor(years[test_idx], dtype=torch.float32).to(self.device)
        )

        return train_data, test_data
    
    # def prepare_arrays(self, images, yields, locations, indices, years, predict_year, time):
    #     train_idx = np.nonzero(years < predict_year)[0]
    #     test_idx = np.nonzero(years == predict_year)[0]

    #     train_images, test_images = self._normalize(images[train_idx], images[test_idx])

    #     print(f'Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}')

    #     # Convert yield data from kg/ha to bushels per acre
    #     yields = convert_kg_per_ha_to_bu_per_acre(yields)

    #     # Standardize yield data
    #     yields, self.yield_mean, self.yield_std = standardize_yield_data(yields)

    #     # Calculate percentage change from previous year
    #     pct_change = np.zeros_like(yields)
    #     for i in range(1, len(yields)):
    #         if years[i] == years[i - 1] + 1:
    #             pct_change[i] = (yields[i] - yields[i - 1]) / yields[i - 1]

    #     # Calculate mean yield per region (state-wise or district-wise)
    #     region_mean_yield = np.zeros_like(yields)
    #     unique_regions = np.unique(locations)
    #     for region in unique_regions:
    #         region_idx = np.where(locations == region)[0]
    #         region_mean_yield[region_idx] = np.mean(yields[region_idx])

    #     # Choose which feature to use (comment out the other)
    #     # Use percentage change from previous year
    #     #additional_feature = pct_change

    #     # Use mean yield per region
    #     additional_feature = region_mean_yield

    #     Data = namedtuple('Data', ['images', 'yields', 'locations', 'indices', 'years', 'additional_feature'])

    #     train_data = Data(
    #         images=torch.as_tensor(train_images[:, :, :time, :], dtype=torch.float32).to(self.device),
    #         yields=torch.as_tensor(yields[train_idx], dtype=torch.float32).unsqueeze(1).to(self.device),
    #         locations=torch.as_tensor(locations[train_idx], dtype=torch.float32).to(self.device),
    #         indices=torch.as_tensor(indices[train_idx], dtype=torch.float32).to(self.device),
    #         years=torch.as_tensor(years[train_idx], dtype=torch.float32).to(self.device),
    #         additional_feature=torch.as_tensor(additional_feature[train_idx], dtype=torch.float32).unsqueeze(1).to(self.device)
    #     )

    #     test_data = Data(
    #         images=torch.as_tensor(test_images[:, :, :time, :], dtype=torch.float32).to(self.device),
    #         yields=torch.as_tensor(yields[test_idx], dtype=torch.float32).unsqueeze(1).to(self.device),
    #         locations=torch.as_tensor(locations[test_idx], dtype=torch.float32).to(self.device),
    #         indices=torch.as_tensor(indices[test_idx], dtype=torch.float32).to(self.device),
    #         years=torch.as_tensor(years[test_idx], dtype=torch.float32).to(self.device),
    #         additional_feature=torch.as_tensor(additional_feature[test_idx], dtype=torch.float32).unsqueeze(1).to(self.device)
    #     )

    #     return train_data, test_data

    
    
    def _normalize(self, train_images, val_images):
        """
        Standardize the images to have a mean of 0 and a standard deviation of 1.
        """
        mean = np.mean(train_images, axis=(0, 2, 3))
        std = np.std(train_images, axis=(0, 2, 3))
        train_images = (train_images.transpose(0, 2, 3, 1) - mean) / std
        train_images = train_images.transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean) / std
        val_images = val_images.transpose(0, 3, 1, 2)

        self.mean = mean
        self.std = std

        return train_images, val_images
    
    def inverse_transform(self, data, data_type='image'):
        """
        Perform inverse transformation of standarized data
        """
        if data_type == 'image':
            return data * self.std + self.mean
        elif data_type == 'yield':
            return data * self.yield_std + self.yield_mean
        
    # def _normalize(self, train_images, val_images):
    #     """
    #     Normalize the images using min-max scaling and mean subtraction.
    #     """
    #     mean = np.mean(train_images, axis=(0, 2, 3))
    #     train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
    #     val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

    #     self.min_val = np.min(train_images, axis=(0, 2, 3), keepdims=True)
    #     self.max_val = np.max(train_images, axis=(0, 2, 3), keepdims=True)

    #     train_images = (train_images - self.min_val) / (self.max_val - self.min_val)
    #     val_images = (val_images - self.min_val) / (self.max_val - self.min_val)

    #     return train_images, val_images

    # def inverse_transform(self, data, data_type='image'):
    #     """
    #     Perform inverse min-max scaling and mean addition to the data.
    #     """
    #     if data_type == 'image':
    #         return data * (self.max_val - self.min_val) + self.min_val
    #     elif data_type == 'yield':
    #         data= data * (self.yield_max - self.yield_min) + self.yield_min
    #         return data
        
    def analyze_results(self, true, pred, pred_gp):
        true = self.inverse_transform(true, data_type='yield')
        pred = self.inverse_transform(pred, data_type='yield')
        if pred_gp is not None:
            pred_gp = self.inverse_transform(pred_gp, data_type='yield')

        # Ensure pred has the same shape as true
        if pred.ndim > 1:
            pred = pred.mean(axis=1)

        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)
        r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
        rmse_t_ha = rmse / 1000  # Convert RMSE to t/ha

        print(f'Without GP: RMSE: {rmse}, ME: {me}, R2: {r2}, RMSE (t/ha): {rmse_t_ha}')

        if pred_gp is not None:
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            r2_gp = 1 - np.sum((true - pred_gp) ** 2) / np.sum((true - np.mean(true)) ** 2)
            rmse_gp_t_ha = rmse_gp / 1000  # Convert RMSE to t/ha

            print(f'With GP: RMSE: {rmse_gp}, ME: {me_gp}, R2: {r2_gp}, RMSE (t/ha): {rmse_gp_t_ha}')
            return rmse, me, r2, rmse_t_ha, rmse_gp, me_gp, r2_gp, rmse_gp_t_ha
        return rmse, me, r2, rmse_t_ha

    def reinitialize_model(self, _=None):
        raise NotImplementedError