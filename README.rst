|test| |codecov| |docs|

.. |test| image:: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/tree/master
    :alt: Test status

.. |codecov| image:: https://img.shields.io/codecov/c/github/Intelligent-Systems-Phystech/ProjectTemplate/master
    :target: https://app.codecov.io/gh/Intelligent-Systems-Phystech/ProjectTemplate
    :alt: Test coverage

.. |docs| image:: https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intelligent-systems-phystech.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Influence of hyperparameters on online aggregation with countable experts
    :Тип научной работы: M1P
    :Автор: Кунин-Богоявленский Сергей Михайлович
    :Научный руководитель: к.ф.-м.н. Зухба Расим Даурович
    :Научный консультант: к.ф.-м.н. Зухба Анастасия Викторовна

Abstract
========

Aggregating forecasts from multiple experts is a valuable method to improve prediction accuracy.
This paper examines the influence of hyperparameters on the performance of the aggregation algorithm for a countable number of experts.
We implement a time series generator with specified properties and an aggregating forecasting model.
We conduct a series of experiments with various hyperparameters of the algorithm and propose a new mixing scheme, used in the algorithm.
The experiments confirm that these hyperparameters have a significant influence on the algorithm's performance.

Аннотация
========

Агрегирование экспертных прогнозов является ценным методом повышения точности предсказаний.
Cтатья исследует влияние гиперпараметров на точность алгоритма агрегирования для счетного числа экспертов.
Реализован генератор временных рядов с заданными свойствами и алгоритм агрегирования экспертных прогнозов.
Проведена серия экспериментов с различными гиперпараметрами алгоритма (такие как cxема смешивания, веса инициализации и др.), a также предложена новая схема смешивания.
Эксперименты подтверждают, что данные гиперпараметры оказывают существенное влияние на точность алгоритма.


Paper latest version can be found here: <https://github.com/intsystems/2024-Project-125/blob/master/paper/KuninBogoiavlenskii2024ExpertsAggregating.pdf>


.. Presentations at conferences on the topic of research
.. ================================================
.. 1. 

.. Software modules developed as part of the study
.. ======================================================
.. 1. A python package *mylib* with all implementation `here <https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/tree/master/src>`_.
.. 2. A code with all experiment visualisation `here <https://github.com/Intelligent-Systems-Phystech/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/Intelligent-Systems-Phystech/ProjectTemplate/blob/master/code/main.ipynb>`_.
