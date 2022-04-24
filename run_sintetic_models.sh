
#for i in 1 2
#do
#  for cat in 'dense' 'avg'
#  do
#    for h2 in 0.05 0.1 0.7 1 7
#    do
#      for depth in 1 2 3
#      do
#        python run_dgi_2_graphs.py --cat=${cat} --h2=${h2} --depth=${depth}
#        python run_dgi_singular_graph.py
#        python run_vgae_singular_graph.py --h2=${h2} --depth=${depth}
#      done
#    done
#  done
#done


#for i in 1 2
#do
#  for h2 in 0.01 0.05 0.1 0.7 1  3 7
#  do
#    for cat in 'dense' 'avg'
#    do
#      for depth in 2 3
#      do
#        python run_hetero_graph.py --cat=${cat} --h2=${h2} --depth=${depth}
#      done
#    done
#  done
#done

for h2 in 0.05 0.5 1 2 10
do
  for depth in 1 2 3
  do
    #python run_dgi_singular_graph.py --h2=${h2} --depth=${depth}
    python run_vgae_singular_graph.py --h2=${h2} --depth=${depth} --loss='mmd'
  done
done