# python format_ycc.py --graph "/data1/competition/icdm/data/icdm2022_session1_edges.csv" \
#         --node  "/data1/competition/icdm/data/icdm2022_session1_nodes.csv" \
#         --label "/data1/competition/icdm/data/session1_label_type.csv" \
#         --storefile "/data1/competition/icdm/data"\
#         --reload True

python format_pyg.py --graph "/data1/competition/icdm/data/icdm2022_session2_edges.csv" \
        --node  "/data1/competition/icdm/data/icdm2022_session2_nodes.csv" \
        --storefile "/home/yinchongchao/pandora/notebooks/competition/icdm/session2"\
        --reload True
        