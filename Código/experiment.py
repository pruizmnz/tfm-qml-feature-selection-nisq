from vqc import VQC_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import umap
import random
import itertools, json
import os 
import pandas as pd
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler, normalize
from datetime import datetime, timedelta
import time
import argparse
import sys

def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    return seed

class Experiment:
    __doc__ = """Esta clase define la configuración del experimento para VQC. Permite crear y ejecutar experimentos con un modelo VQC."""

    def __init__(self, models_config = {},models_path=None,results_path=None,name="experiment",seed=42):
        self.models_config = models_config
        self.finished = False
        self.current_step = None
        self.models_path = models_path if models_path is not None else './models'
        self.results_path = results_path if results_path is not None else './results'
        
        self.seed = seed
        self.loss_history = []
        self.val_acc_history = []

        self.exp_name = name
    
    def training_callback(self,weights,loss):
        """
        Calback function used during VQC training.

        weights : np.ndarray

        loss : float
            The current value of the objective function being optimized.
        """
        self.loss_history.append(loss)
    
        if len(self.loss_history) % 3 == 0: 
            if self.use_val:
                val_x = self.val_x[self.current_ds]
                raw_predictions = self.current_model.vqc.neural_network.forward(val_x, weights)
                val_predictions = np.argmax(raw_predictions, axis=1)
                #self.current_model.vqc.weights = weights
                #val_predictions = self.current_model.vqc.predict(self.val_x)
                acc_val = accuracy_score(self.val_y, val_predictions)
                self.val_acc_history.append(acc_val)
                print(f"Iteration {len(self.loss_history)}: loss = {loss} , validation accuracy = {acc_val}")
            print(f"Iteration {len(self.loss_history)}: loss = {loss}")
        

    def evaluate_model(self,model,test_X,test_y): 
        predictions = model.predict(test_X)

        if np.unique(test_y).size ==2:
            average = 'binary'
        else:
            average = 'weighted'

        acc = accuracy_score(test_y, predictions)
        prec = precision_score(test_y, predictions, average=average)
        rec = recall_score(test_y, predictions, average=average)
        f1 = f1_score(test_y,predictions,average=average)
        cm = confusion_matrix(test_y, predictions)

        return acc,prec,rec,f1,cm
    
    def featureSelection(self,config,train,test,val=None):

        featureSelectionConf = config.get("featureSelection",{"method":"none"})
        method = featureSelectionConf.get("method")

        if method == "none":
            print("Sin aplicar Reducción de características")
            val_ret = [pd.DataFrame(val)] if val is not None else None
            return ([pd.DataFrame(train)], val_ret, [pd.DataFrame(test)])
        
        num_features_list = featureSelectionConf.get("numFeatures", [len(train.columns)])

        trainList = []
        valList = [] if val is not None else None
        testList = []

        if method == "manual":
            selection_order = featureSelectionConf.get("manual_selection_order",list(range(len(train.columns))))
            featureNames = train.columns.tolist()

            featureOrder = sorted(zip(featureNames, selection_order), key=lambda x: x[1])
            sortedFeatures = [f for f, r in featureOrder]

            feat_to_select = featureSelectionConf.get("numFeatures",[len(train.columns)])

            for k in num_features_list:
                selected = sortedFeatures[:k]
                trainList.append(train[selected])
                testList.append(test[selected])
                if val is not None:
                    valList.append(val[selected])
        
            return (trainList, valList, testList)
        
        for n_components in num_features_list:
            reducer = None
            # 1. Instanciamos el reductor
            if method == "PCA":
                reducer = PCA(n_components=n_components,random_state=self.seed)
            
            elif method == "UMAP":
                reducer = umap.UMAP(n_components=n_components,random_state=self.seed)
                
            elif method == "autoencoder":
                pass
            if reducer is not None:
                # 2. FIT: Solo en Train
                reducer.fit(train)

                train_trans = pd.DataFrame(reducer.transform(train))
                test_trans  = pd.DataFrame(reducer.transform(test))
                
                trainList.append(train_trans)
                testList.append(test_trans)

                if val is not None:
                    val_trans = pd.DataFrame(reducer.transform(val))
                    valList.append(val_trans)
            
            if not trainList: 
                print(f"Advertencia: Método '{method}' no reconocido o sin resultados. Devolviendo datos originales.")
                val_ret = [val] if val is not None else None
                return ([train], val_ret, [test])

        return (trainList, valList, testList)
    
        
    def normalization(self,config,train,test,val=None):
        
        #De momento por defecto a 0-Pi
        norm_config = config.get("normalization", {})
        type = norm_config.get("type", "range")
        

        train_norm = []
        test_norm = []
        val_norm = [] if val is not None else None
        if type == "none":
            return (train, test, val)
        if type == "amplitude":
            df_train = train[0].copy()
            df_test = test[0].copy()
            df_val = val[0].copy() if val is not None else None

            current_features = train[0].shape[1]
            next_pow2 = 2**int(np.ceil(np.log2(current_features)))
            padding_needed = next_pow2 - current_features

            if padding_needed >0:
                for p in range(padding_needed):
                    pad_col = f'pad_{p}'
                    df_train[pad_col] = 0.0
                    df_test[pad_col] = 0.0
                    if df_val is not None:
                            df_val[pad_col] = 0.0
            
            train_norm_arr = normalize(df_train.values,norm='l2')
            test_norm_arr = normalize(df_test.values,norm='l2')
            
            train_norm.append(pd.DataFrame(train_norm_arr, columns=df_train.columns))
            test_norm.append(pd.DataFrame(test_norm_arr, columns=df_test.columns))
                
            if df_val is not None:
                val_norm_arr = normalize(df_val.values, norm='l2')
                val_norm.append(pd.DataFrame(val_norm_arr, columns=df_val.columns))
            
        if type =="range":
            feature_range = norm_config.get("feature_range", (0, np.pi))
            for i in range(len(train)):
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.fit(train[i])

                train_norm.append(pd.DataFrame(scaler.transform(train[i]),columns=train[i].columns))
                test_norm.append(pd.DataFrame(scaler.transform(test[i]),columns=test[i].columns))
                if val is not None:
                    val_norm.append(pd.DataFrame(scaler.transform(val[i]),columns=val[i].columns))

        return train_norm, test_norm, val_norm

    def prepareSplit(self,config,dataset):
        val_method = config.get("val_method",{"type":"hold_out","test_size":0.2})
        use_val = config.get("use_val_set","N").upper() == 'Y'
        self.use_val = use_val
        val = None
        if val_method.get("type") =="hold_out":
            train, test = train_test_split(dataset, test_size=val_method.get("test_size", 0.2), random_state=self.seed)
            
            if use_val:
                train, val = train_test_split(train, test_size=0.1, random_state=self.seed)

        elif val_method.get("type") == "stratified_ho":
            train, test = train_test_split(dataset, test_size=val_method.get("test_size", 0.2), random_state=self.seed, stratify=dataset.iloc[:, -1])
            if use_val:
                train, val = train_test_split(train, test_size=0.1, random_state=self.seed, stratify=train.iloc[:, -1])

        elif val_method.get("type") == "manual_split":
            train = dataset[0]
            if len(dataset) == 3:
                val = dataset[1]
                test = dataset[2]
            else:
                test = dataset[1]
                if use_val:
                    train, val = train_test_split(train, test_size=0.2, random_state=self.seed, stratify=train.iloc[:, -1])
        else:
            train, test = train_test_split(dataset, test_size=0.2, random_state=self.seed)
            val = None

        train_y = train.iloc[:,-1]
        test_y = test.iloc[:, -1]
        val_y = val.iloc[:, -1] if val is not None else None

        print(f"Config FS:{config}")
        train,val,test = self.featureSelection(config=config,
                                               train=train.iloc[:,:-1],
                                               test=test.iloc[:,:-1],
                                               val=val.iloc[:, :-1] if val is not None else None)
        
        train,test,val = self.normalization(config=config,
                                            train=train,
                                            test=test,
                                            val=val)
        
        return train, val, test, train_y, val_y, test_y


    def run(self,dataset,config,progress_path=None):

        # Get train, test and validation datasets
        train, val, test, train_y, val_y, test_y = self.prepareSplit(config,dataset)
    
        self.val_x = val
        self.val_y = val_y

        #Compruebo antes de empezar a crear los modelos que existe un ficero json donde almacenar los resultados
        
        #Compruebo que existe la carpeta results
        os.makedirs(self.results_path,exist_ok=True)
        savefile_name = f"{self.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
        save_path = os.path.join(self.results_path,savefile_name)
        #Creo el fichero
        with open(save_path,'w') as f:
            json.dump({"models_configurations":{}},f)
        
        for params in ParameterGrid(self.models_config):
            for i in range(len(train)):
                self.current_ds = i
                params["num_features"] = len(train[i].columns)
                params["num_classes"] = train_y.nunique()
                print(params)

                #Reinitialize the loss history
                self.loss_history = []
                self.val_acc_history = []

                model = VQC_model(config=params,t_callback=self.training_callback)
                
                self.current_model = model

                print(f"Model created with{len(train[i].columns)} features")
                #print(f"Repeticiones Ansatz:{model.vqc.ansatz.reps}")
                print("Starting training")
                start = time.time()
                model.vqc.fit(train[i],train_y.values)
                elapsed = time.time() - start
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                print(f"Training time:{elapsed_str}")
                #Evalúo el modelo y lo guardo en la carpeta 'models'
                acc,prec,rec,f1,cm = self.evaluate_model(model=model.vqc,test_X=test[i],test_y=test_y)
                print(f"Resultado!: acc:{acc}, prec:{prec}, rec:{rec},f1:{f1},{cm}")
                model_name = f"VQC_{train[i].size}_{datetime.timestamp(datetime.now())}.model"
                model.vqc.save(f"./models/{model_name}")

                #Almaceno los resultados del modelo
                with open(save_path,'r') as f:
                    savefile = json.load(f)
                
                savefile[model_name]={"acc":acc,"prec":prec,"recall":rec,"f1":f1,"confusion_matrix":cm.tolist(),"training_time":elapsed_str,"loss_history":self.loss_history,"val_history":self.val_acc_history} 
                savefile["models_configurations"][model_name] = params

                with open(save_path,'w') as f:
                    json.dump(savefile,f,indent=4)

                print(f"Almacenados los resultados del modelo: {model_name}")

        return True
    

if __name__=="__main__":

    seed = set_global_seed(42)

    # Args
    parser = argparse.ArgumentParser(description="Lanzador de Experimentos QML para el Clúster")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--models", type=str, help="Models configuration path")
    parser.add_argument("--training", type=str, help="Training configuration path")
    parser.add_argument("--train", type=str, help="Train path")
    parser.add_argument("--test", type=str, help="Test path")

    args = parser.parse_args()
    provided_args = [args.name, args.models, args.training, args.train, args.test]

    if all(provided_args):
        # Se han pasado todos los argumentos y se procede con la ejecución configurada
        print("-> Running experiment on passed arguments")
        exp_name = args.name
        models_path = args.models
        training_path = args.training
        train_path = args.train
        test_path = args.test

    elif not any(provided_args):
        # Si no se pasa ningún parámetro se realiza una ejecución por defecto
        print("-> Running experiment on hardcoded configuration")
        exp_name = "Experimento_PCA_Even_ZZ_RA2L_MNIST01"
        # Model_Pauli_RA2L_COBYLA
        # Model_ZZ_RA2L_COBYLA
        models_path = './config_files/FinalExperiments/Model_ZZ_RA2L_COBYLA.json'

        training_path = './config_files/FinalExperiments/PCA_even_val.json'

        train_path = './datasets/mnist/mnist_subset_train_balanced_01.csv'
        test_path = './datasets/mnist/mnist_subset_test_balanced_01.csv'

    else:
        # Error si falta algún parámetro de configuración
        parser.error("Every argument (--name, --models, --training, --train, --test) or non must be pased.")

    
    with open(models_path,'r') as file:
        models_config = json.load(file)
    
    with open(training_path) as file:
        training_config = json.load(file)

    #Manual split
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    dataset = [train,test]

    exp = Experiment(models_config=models_config,name=exp_name,seed=seed)

    print("Starting Experiment:", datetime.now().strftime("%H:%M:%S"))
    exp.run(dataset=dataset,config=training_config)