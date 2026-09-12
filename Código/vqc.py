from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA,SLSQP
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.circuit.library import RealAmplitudes, EfficientSU2 , ExcitationPreserving, TwoLocal


def createFeatureMap(type='zz',feature_dimension=5,reps=2,entanglement = 'full',paulis=['Z']):
        
        entanglement = entanglement if entanglement in ['full','linear','reverse_linear','circular','sca'] else 'full'
        reps = reps if isinstance(reps,int) and (1<=reps<=10) else 2
        feature_dimension = feature_dimension if isinstance(feature_dimension,int) else 5

        if type == 'zz':
            fMap = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement=entanglement)
        elif type =='pauli':
            fMap = PauliFeatureMap(feature_dimension=feature_dimension,reps=reps,entanglement=entanglement,paulis=paulis)
        elif type =='amplitude':
            fMap = RawFeatureVector(feature_dimension=feature_dimension)
        else:
            fMap = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement=entanglement)
            
        return fMap


def createAnsatz(type='real_amplitudes',num_qubits=5,reps=2,entanglement = 'full',mode='iswap'):
    
    entanglement = entanglement if entanglement in ['full','linear','reverse_linear','circular','sca'] else 'full'
    reps = reps if isinstance(reps,int) and (1<=reps<=10) else 2
    num_qubits = num_qubits if isinstance(num_qubits,int) and (2<=num_qubits<=10) else 5

    if type == 'real_amplitudes':
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    elif type == 'efficient_su2':
        ansatz = EfficientSU2(num_qubits=num_qubits,reps=reps,entanglement=entanglement) 
    elif type=='excitation_preserving':
        ansatz = ExcitationPreserving(num_qubits=num_qubits,reps=reps,entanglement=entanglement,mode=mode)
    elif type == 'two_local':
        ansatz = TwoLocal(num_qubits=num_qubits,reps=reps,entanglement=entanglement)
    else:
        #Default ansatz
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
        
    return ansatz



def createOptimizer(type='slsqp',maxiter=1000):
    
    
    if type == 'slsqp':
        optimizer = SLSQP(maxiter=maxiter)
    if type == 'cobyla':
        optimizer = COBYLA(maxiter=maxiter)
    else:
        optimizer = SLSQP(maxiter=maxiter)
        
    return optimizer


class VQC_model:
    __doc__ ="""Esta clase facilita la creación del modelo VQC a partir de un fichero de configuración."""

    def __init__(self,feature_map=None, ansatz=None, optimizer=None,config={},t_callback=None):
        self.trained = False
        self.config = config
        self.output_shape = config.get('num_classes',2)
        if feature_map is None:
            fMapConf = config.get('feature_map', {})
            self.feature_map = createFeatureMap(type=fMapConf.get('type', 'zz'),
                                                feature_dimension=config.get('num_features', 5),
                                                reps=fMapConf.get('reps', 2),
                                                entanglement=fMapConf.get('entanglement', 'full'))
        else:
            self.feature_map = feature_map

        if ansatz is None:
            ansatzConf = config.get('ansatz', {})
            self.ansatz = createAnsatz(type=ansatzConf.get('type', 'real_amplitudes'),
                                       num_qubits=self.feature_map.num_qubits,
                                       reps=ansatzConf.get('reps', 2),
                                       entanglement=ansatzConf.get('entanglement', 'full'))
        else:
            self.ansatz = ansatz

        if optimizer is None:
            optimizerConf = config.get('optimizer', {})
            self.optimizer = createOptimizer(type=optimizerConf.get('type', 'slsqp'),
                                             maxiter=optimizerConf.get('maxiter', 1000))
        else:
            self.optimizer = optimizer

        self.vqc = VQC(feature_map=self.feature_map,
                      ansatz=self.ansatz,
                      optimizer=self.optimizer,
                      output_shape=self.output_shape,
                      callback=t_callback)
        