#Declaramos las librerias que utilizaremos en el programa
import tensorflow as tf, sys


##Esta linea es la que lee la imagen del argumento
image_path = sys.argv[1]


# Carga las etiquetas
imagen_datos = tf.gfile.FastGFile(image_path, 'rb').read()
etiqueta_lineas = [line.rstrip() for line
    in tf.gfile.GFile("retrained_labels.txt")]
 
# Con esto se cargan los grafos de la red
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Con esto empieza a decir que tipo de animal es y nos da las predicciones.
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predicciones = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': imagen_datos})
    
    top_k = predicciones[0].argsort()[-len(predicciones[0]):][::-1]
    for node_id in top_k:
        human_string = etiqueta_lineas[node_id]
        resultados = predicciones[0][node_id]
        print('%s (score = %.5f)' % (human_string, resultados))
