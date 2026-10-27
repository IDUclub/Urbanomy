Urbanomy
=========================================================
.. logo-start

.. figure:: docs/img/urbanomy.png
   :alt: Urbanomy

.. logo-end

|TestsStatus| |DocsStatus| |PythonVersion|

.. readme-start

Overview
--------
Urbanomy is a research toolkit from the Institute of Design and Urban Studies (IDU) for assessing land parcels from spatial, economic, and scenario perspectives. The library bundles geospatial feature engineering, CatBoost-based land-price modelling, multi-criteria investment scoring, and socio-economic ripple analysis into one reproducible codebase.

Highlights
----------
- **Land-value modelling** – build spatial lag features, estimate land prices, and visualise impacts with pre-trained CatBoost models (`urbanomy.methods.land_value_modeling <urbanomy/methods/land_value_modeling>`_).
- **Scenario planning** – modify individual blocks, recompute indicators, and plot price deltas for what-if cases (`urbanomy.methods.land_value_modeling.scenario_modification <urbanomy/methods/land_value_modeling/scenario_modification.py>`_).
- **Investment attractiveness** – aggregate cashflows, NPV/IRR/ROI metrics, and weighted spatial potential into a combined INV score (`urbanomy.methods.investment_potential <urbanomy/methods/investment_potential>`_).
- **Socio-economic footprint** – quantify fiscal and employment effects by combining one-off construction contributions with operating-phase impacts using `SEREstimator <urbanomy/methods/socio_economic_indicators/sei_calculate.py>`_.
- **Investment input validation** – preparation and validation of scenario tables (`urbanomy.methods.investment_potential.input <urbanomy/methods/investment_potential/input.py>`_).

Installation
------------
Urbanomy requires Python 3.10 and system libraries for GeoPandas (GEOS, GDAL, PROJ). We recommend working inside a virtual environment.

.. code-block:: bash

   git clone https://github.com/iduclub/Urbanomy.git
   cd Urbanomy
   pip install -e .

Optional dependency groups:

- ``pip install -e '.[dev]'`` – formatting, linting, pre-commit hooks.
- ``pip install -e '.[test]'`` – pytest and coverage tooling.
- ``pip install -e '.[docs]'`` – Sphinx stack for building the documentation.
- ``pip install -e '.[llm]'`` – LangChain 1.x, LangGraph, and the OpenAI-compatible LLM client.

Quickstart
----------
Land-value modelling
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   import geopandas as gpd
   from catboost import CatBoostRegressor
   from urbanomy.methods.land_value_modeling import (
       LandDataPreparator,
       LandPriceEstimator,
       plot_land_price_maps,
   )

   scenario = gpd.read_file("data/scenario_blocks.geojson")
   context = gpd.read_file("data/context_blocks.geojson")
   # Accessibility matrix is computed automatically assuming 5 km/h travel speed.
   preparator = LandDataPreparator(
       scenario_blocks_source=scenario,
       context_blocks_source=context,
   )
   prepared_blocks = preparator.prepare()

   model = CatBoostRegressor()
   model.load_model("models/land_price.cbm")

   estimator = LandPriceEstimator(model=model, blocks=prepared_blocks)
   price_map = estimator.predict()
   plot_land_price_maps(blocks_pred=price_map)

.. figure:: docs/img/1_pricing_model.png
   :alt: Training the land-price model with spatial lags
   :width: 75%

   Training the land-price model and generating spatial lag features

.. figure:: docs/img/3_land_price_modeling_1.png
   :alt: Price change map after scenario modification
   :width: 75%

   Price change map after applying the scenario

.. figure:: docs/img/3_land_price_modeling_2.png
   :alt: Comparing investments and prices for scenario and context blocks
   :width: 75%

   Comparing investments and prices for scenario and context blocks

Investment attractiveness
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from urbanomy.methods.investment_potential import InvestmentAttractivenessAnalyzer

   # blocks_full_value contains the predicted prices before and after the scenario.
   investment_block = blocks_full_value.loc[blocks_full_value["id"] == target_id].copy()
   investment_block["is_project"] = True

   # Uses the reference benchmarks, including demolition costs.
   analyzer = InvestmentAttractivenessAnalyzer()
   project_summary = analyzer.calculate_investment_metrics(investment_block)

For scenario analysis, pass the valued layer containing both ``land_value``
(the prediction after the change) and ``land_value_before``. The investment
report's ``land_value`` is the acquisition cost before the change. Demolition
uses ``build_floor_area_before * cost_demolition`` and is charged once in the
initial cash flow, so it affects both investment needs and NPV.

.. figure:: docs/img/investment_metrics.png
   :alt: Investment metrics dashboard with INV index
   :width: 75%

   Investment metrics dashboard with the final INV index

Socio-economic effects
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   import pandas as gpd
   from urbanomy.methods.ser import SEREstimator

   ser = SEREstimator(
       {
           "population": 52_000,
           "employment_base": 21_000,
           "avg_wage_base": 70_000,
       }
   )

   project_blocks = gpd.read_file('./data/test/blocks_investment.geojson')
   ser_table = ser.compute(project_blocks)
   print(ser_table)

LLM-assisted optimization
~~~~~~~~~~~~~~~~~~~~~~~~~
The agent implementation lives in ``urbanomy.methods.agent`` and replaces the former
``urbanomy.methods.agent_interface`` package. Configure ``OPENAI_API_KEY`` in the
environment or a local ``.env`` file. For an OpenAI-compatible endpoint, also set
``CHAT_URL`` and ``CHAT_MODEL``; ``FP2MP_CHAT_URL``, ``FP2MP_API_KEY``, and
``FP2MP_MODEL`` take precedence over the corresponding ``CHAT_*`` settings.

.. code-block:: python

   from urbanomy.methods.agent import SingleAgentBaseline, init_llm
   from urbanomy.methods.land_value_modeling import (
       Evaluation,
       run_nsga2_with_strategic_alignment,
   )

   baseline = SingleAgentBaseline(
       llm=init_llm(),
       output_schema=Evaluation,
   )
   # Pass baseline and a non-empty prompt to run_nsga2_with_strategic_alignment,
   # together with blocks, model, estimator_kwargs, constraints, and target_id.
   # The third objective is an LLM score between 0 and 1.

``build_pareto_front_dataframe`` returns the target ID, repaired parameters, land
values, investor NPV, and an ``llm score`` column for three-objective results.

Repeated scenario evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reuse a ``LandPriceEstimator`` with ``estimator.predict(blocks=scenario_blocks)``
when evaluating several scenarios. Spatial weights are reused when geometry,
index order, and CRS match the constructor's dataset; changed layouts rebuild
the weights. Numeric features and spatial lags are recalculated for every
scenario. ``DistrictProblem`` reuses the estimator automatically during NSGA-II.

Training and prediction share the same spatial-lag implementation. Investment
potential scores are calculated from the original attributes, so writing one
land-use score cannot change the inputs for the next score. Input data is
preserved during investment preparation.

Notebook-facing imports remain available at their existing paths. Method modules and
default LLM/embedding clients are loaded on demand; importing ``init_llm`` does
not construct clients or require credentials.

Code organization
~~~~~~~~~~~~~~~~~
The public package exports used by the notebooks remain the entry points.
Implementation modules are organized by responsibility:

- ``agent/llms.py`` owns environment settings and client construction;
  ``agent/single_agent.py`` contains the single-agent graph and its wrapper.
- ``investment_potential/input.py`` prepares and validates scenario tables;
  ``cashflows.py`` calculates financial results; ``investment_metrics.py``
  assembles the reports exposed by ``InvestmentAttractivenessAnalyzer``.
- ``land_value_modeling/pareto.py`` builds result tables for NSGA-II with two
  or three objectives. LLM scoring during optimization is implemented in
  ``strategic_alignment.py`` and uses the single-agent baseline.
- ``scenario_modification.py`` changes block indicators;
  ``scenario_summary.py`` computes and formats summaries;
  ``scenario_visualization.py`` renders impact maps.
- ``urbanomy/utils.py`` contains dependency-free payload conversions.

Internal paths changed: investment input preparation moved from
``urbanomy.utils.investment_input`` to
``urbanomy.methods.investment_potential.input``. Prefer the unchanged public
import ``from urbanomy.methods.investment_potential import prepare_investment_input``.
Single-agent implementation files were consolidated into
``urbanomy.methods.agent.single_agent``; Pareto tables are exported from
``urbanomy.methods.land_value_modeling`` as before.

The former ``pareto_llm_selector`` voting workflow and its dedicated tools and
models were removed after checking all notebooks in ``examples/``. The active
LLM workflow scores candidates during NSGA-II optimization with prompt 3.
The notebooks also use training, data preparation, land-price prediction,
scenario maps, cash-flow analysis, and socio-economic indicators. ``Agent`` and
``LandUseScoreAnalyzer`` have no direct calls in the notebooks and remain
additional public APIs.

Datasets and notebooks
----------------------
Reproducible pipelines live in ``examples/``:

- `examples/NSGA2_LLM.ipynb <examples/NSGA2_LLM.ipynb>`_ – NSGA-II optimization with a structured LLM scorer. Install ``.[llm,dev]`` and run from ``examples/``. It expects ``data/blocks_agg_with_indicators.geojson`` and ``data/catboost_land_value_no_services.cbm``. Its final comparison also requires precomputed ``nsga_2_without_llm/no_llm_pareto_front_<seed>.jsonl`` results. These experiment assets are supplied separately.
- `1_block_cadastral_valuation.ipynb <examples/land_value_modeling/1_block_cadastral_valuation.ipynb>`_ – aggregate cadastral parcel values to urban blocks.
- `2_cadastral_parcel_features.ipynb <examples/land_value_modeling/2_cadastral_parcel_features.ipynb>`_ – enrich cadastral parcels with land use, building indicators, morphotypes, and accessibility.
- `3_land_price_classification.ipynb <examples/land_value_modeling/3_land_price_classification.ipynb>`_ – compare price grouping and CatBoost classification approaches.
- `4_land_price_model_training.ipynb <examples/land_value_modeling/4_land_price_model_training.ipynb>`_ – train and evaluate land-price regression models with spatial lags.
- `5_scenario_context_preparation.ipynb <examples/land_value_modeling/5_scenario_context_preparation.ipynb>`_ – prepare scenario and context blocks with ``LandDataPreparator``.
- `examples/gatchina_scenarios.ipynb <examples/gatchina_scenarios.ipynb>`_ – compare NSGA-II optimization with and without LLM scoring for Gatchina.
- `examples/pipeline.ipynb <examples/pipeline.ipynb>`_ – geospatial preparation, scenario valuation, investment metrics, and socio-economic reporting.
- `examples/pipeline_prediction_data.ipynb <examples/pipeline_prediction_data.ipynb>`_ – prepare input blocks for land-price prediction.

All notebooks rely on sample assets committed to ``examples/data`` (GeoJSON blocks, accessibility matrices, CatBoost weights, and prepared pickle files). When running the notebooks from another working directory, point the loaders to that folder or provide equivalent datasets in the same structure.

Data repository
---------------
The data required by the library is available in a dedicated repository: `Urbanomy-data <https://github.com/Mvin8/Urbanomy-data.git>`_.

Documentation
-------------
The latest documentation is published automatically from ``main``:

https://iduclub.github.io/Urbanomy/

Build the docs locally with:

.. code-block:: bash

   pip install -e '.[docs]'
   sphinx-build docs/source docs/build

Development
-----------
We use the supplied Makefile shortcuts:

.. code-block:: bash

   make install-dev   # install editable package with dev tools
   make lint          # pylint on urbanomy
   make test          # pytest tests/
   make install-docs  # editable install with docs extras

Before opening a pull request:

1. Format imports with ``isort`` and code with ``black`` (``make format``).
2. Ensure ``make test`` passes.
3. Run ``sphinx-build`` if documentation is affected.

License
-------
Urbanomy is released under the BSD-3-Clause license. See ``LICENSE`` for the full text.

Contact
-------
Maintainer: Maksim Natykin (``mvin@itmo.ru``)  
Issues & feature requests: https://github.com/iduclub/Urbanomy/issues

.. |TestsStatus| image:: https://github.com/iduclub/Urbanomy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/iduclub/Urbanomy/actions/workflows/tests.yml
   :alt: Tests status

.. |DocsStatus| image:: https://github.com/iduclub/Urbanomy/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/iduclub/Urbanomy/actions/workflows/documentation.yml
   :alt: Documentation build status

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
   :alt: Supported Python version
