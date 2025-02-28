# Performance tracking

This repository is used to track the performance and energy consumption of different algorithms in terms of hardware usage and inference time. These comparisons are made using data fetched on InfluxDB, a time-series database, and visualized on Grafana, a data visualization tool.  

```bash
.
├── config
│   ├── get_metrics.sh
│   ├── grafana_dashboard_template.json
│   ├── README.md
│   ├── screenshots_config
│   │   ├── grafana_import_dashb1.png
│   │   ├── grafana_import_dashb2.png
│   │   ├── grafana_import_dashb3.png
│   │   ├── grafana_select_influx.png
│   │   ├── grafana_set_datasource1.png
│   │   ├── grafana_set_datasource2.png
│   │   └── grafana_welcome.png
│   ├── smart-switch
│   │   ├── add-device-ha.png
│   │   ├── config-homeassist.png
│   │   ├── enable-stats.png
│   │   ├── first-data.png
│   │   ├── new-entry.png
│   │   ├── pipeline.png
│   │   ├── smart-start.png
│   │   ├── smartswitch7.jpg
│   │   └── zstick7.jpg
│   └── tig
│       ├── docker-compose.yml
│       └── telegraf
│           └── telegraf.conf
├── experiments
│   ├── conso
│   │   ├── analyse_stats.py
│   │   ├── get_conso.py
│   │   ├── get_stats.py
│   │   ├── query_influx.sh
│   │   ├── simulation_metrics_exec.sh
│   │   ├── stats_summary_blob.py
│   │   ├── stats_summary_deep.py
│   │   └── stats_summary.py
│   ├── conso_change
│   │   ├── cd_sklearn_pair_var.py
│   │   ├── change-detection.py
│   │   ├── functions.py
│   │   ├── get_perf.py
│   │   ├── helpers
│   │   │   └── multivariate_images_tool.py
│   │   ├── main.py
│   │   ├── param_change_glrt_2images.yaml
│   │   ├── param_change_interm.yaml
│   │   ├── param_change_logdiff_2images.yaml
│   │   ├── param_change_robust_2images.yaml
│   │   ├── param_change_robust_test.yaml
│   │   ├── param_change_robust.yaml
│   │   └── param_change.yaml
│   ├── conso_classif_deep
│   │   ├── classif_deep.py
│   │   ├── get_perf.py
│   │   ├── get_scores.py
│   │   ├── param_classif_deep_Inception.yaml
│   │   ├── param_classif_deep_SCNN_10.yaml
│   │   ├── param_classif_deep_SCNN_strat.yaml
│   │   ├── param_classif_deep_SCNN.yaml
│   │   ├── param_classif_deep_test.yaml
│   │   ├── param_classif_deep.yaml
│   │   ├── read_event.py
│   │   ├── read_events.py
│   │   └── simulation_metrics_exec.sh
│   └── conso_clustering
│       ├── clustering_blob.py
│       ├── clustering.py
│       ├── get_perf_blob.py
│       ├── get_perf.py
│       ├── helpers
│       │   └── processing_helpers.py
│       ├── param_clustering_blob.yaml
│       ├── param_clustering_interm.yaml
│       ├── param_clustering_test.yaml
│       ├── param_clustering.yaml
│       ├── plot_clustering.py
│       ├── utils_clustering_blob.py
│       └── utils_clustering.py
├── plot_usage.py
├── README.md
└── simulation_metrics_exec.sh
```