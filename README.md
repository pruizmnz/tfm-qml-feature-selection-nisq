# Hacia un Aprendizaje Automático Cuántico adaptado para NISQ: estudio de modelos híbridos asistidos por selección de características

---

## Memoria

* 📕 **Memoria del TFM (PDF):** [Descargar / Ver memoria completa](TFM_PabloRuiz.pdf)

---

## Autoría

* **Autor:** Pablo Ruiz Muñoz
* **Tutores:** Dr. Gabriel Jesús Luque Polo y Dr. Zakaria Abdelmoiz Dahi 
* **Programa:** Máster Universitario en Ingeniería Informática
* **Institución:** Universidad de Málaga
* **Fecha:** Junio 2026

---

## Abstract

La computación cuántica es un área de trabajo en auge en los últimos años.
Actualmente nos encontramos en la era NISQ, donde los ordenadores cuánticos
disponen de un número reducido de cúbits y son susceptibles al ruido afectando a
la precisión de las operaciones realizadas. Bajo este contexto, existe una necesidad
de demostrar la viabilidad práctica del aprendizaje automático cuántico. Por ello y
ante la dificultad de codificar conjuntos de datos de alta dimensionalidad sobre
circuitos con un número reducidos cúbits, en este proyecto se propone el uso de
modelos híbridos asistidos por la selección de características. Con este propósito
se ha desarrollado un marco de experimentación que integra la creación de un
modelo híbrido clásico-cuántico de forma parametrizable, permitiendo la ejecución
de lotes de experimentación mediante ficheros de configuración. Usando este
entorno de experimentación se ha evaluado el efecto de tres enfoques de reducción
(PCA, UMAP y Autoencoders), además de distintas estrategias de codificación
cuántica y configuraciones de feature map de un modelo VQC, empleando los
conjuntos de datos MNIST y FMNIST. Los resultados experimentales demuestran
el efecto positivo de la reducción de dimensionalidad aplicada. Esto habilita el
procesamiento de conjuntos de datos sobre circuitos reducidos y eficientes, que de
otro modo serían prohibitivos en hardware cuántico actual.
