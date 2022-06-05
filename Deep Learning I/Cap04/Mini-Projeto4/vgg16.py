########################################################################
#
# O modelo pré-treinado VGG16 para TensorFlow.
#
# Implementado no Python 3.7 com TensorFlow 2.1
#
########################################################################

import numpy as np
import tensorflow as tf
import download
import os

########################################################################
# Vários diretórios e nomes de arquivos.

# Os nomes das classes estão disponíveis na seguinte URL:
# http://datascienceacademy.com.br/blog/dsa/models/synset.txt

# URL da Internet para o arquivo com o modelo VGG16.
# Note que isso pode mudar no futuro e precisará ser atualizado.
data_url = "http://datascienceacademy.com.br/blog/dsa/models/vgg16.tfmodel"

# Diretório para armazenar os dados baixados.
data_dir = "vgg16/"

# Arquivo contendo a definição do grafo TensorFlow. 
path_graph_def = "vgg16.tfmodel"

########################################################################


def maybe_download():
    """
    Baixa o modelo VGG16 da internet se ainda não
    existe no data_dir. ATENÇÃO! O arquivo tem cerca de 550 MB.
    """

    print("Downloading VGG16 Model ...")

    # O arquivo na internet não é armazenado em um formato comprimido.
    # Esta função não deve extrair o arquivo quando não tiver
    # um arquivo relevante - extensões como .zip ou .tar.gz
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class VGG16:
    """
    O modelo VGG16 é uma Deep Neural Network que já foi treinada para classificar imagens em 1000 categorias diferentes.

    Quando você cria uma nova instância desta classe, o modelo VGG16 será carregado e pode ser usado imediatamente sem treinamento.
    """

    # Nome do tensor para alimentar a imagem de entrada.
    tensor_name_input_image = "images:0"

    # Nomes dos tensores para os valores aleatórios de Dropout ..
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Nomes para as camadas convolucionais no modelo para uso em Transferência de Estilo.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        # Agora carregamos o modelo a partir do arquivo. 

        # Cria um novo grafo computacional TensorFlow.
        self.graph = tf.Graph()

        # Define o novo grafo como o padrão.
        with self.graph.as_default():

            # Os grafos TensorFlow são salvos no disco como chamados Buffers de protocolo
            # Proto-bufs que é um formato de arquivo que funciona em múltiplos
            # Plataformas. Neste caso, ele é salvo como um arquivo binário.

            # Abre o arquivo gráfico-def para leitura binária.
            path = os.path.join(data_dir, path_graph_def)
            with tf.compat.v1.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Obtém referências dos tensores para as camadas comumente usadas.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def get_layer_tensors(self, layer_ids):
        """
        Devolve uma lista de referências aos tensores para as camadas com os ID fornecidos.
        """

        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        """
        Retorna uma lista de nomes para as camadas com os ID fornecidos.
        """

        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        """
        Retorna uma lista de todas as camadas (operações) no grafo.
        A lista pode ser filtrada para nomes que começam com a string fornecida.
        """

        # Obtém uma lista dos nomes para todas as camadas (operações) no grafo.
        names = [op.name for op in self.graph.get_operations()]

        # Filtra a lista de nomes para que possamos obter aqueles que começam com a string dada.
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def create_feed_dict(self, image):
        """
        Cria e retorna um feed-dict com uma imagem.

        :param image:
            A imagem de entrada é uma matriz de 3-dim que já está decodificada.
            Os pixels DEVEM ser valores entre 0 e 255 (flutuante ou int).

        :return:
            Dict para alimentar o grafo no TensorFlow.
        """

        # Expandir a matriz de 3-dim para 4-dim, antecipando uma dimensão "vazia".
        # Isso ocorre porque apenas estamos alimentando uma única imagem, mas
        # o modelo VGG16 foi construído para receber várias imagens como entrada.
        image = np.expand_dims(image, axis=0)

        if False:
            # No código original usando este modelo VGG16, os valores aleatórios
            # para o Dropout são fixados em 1.0.
            # Experimentos sugerem que não parece ser importante para
            # Transferência de estilo, e isso causa um erro com uma GPU.
            dropout_fix = 1.0

            # Cria feed-dict para inserir dados no TensorFlow.
            feed_dict = {self.tensor_name_input_image: image,
                         self.tensor_name_dropout: [[dropout_fix]],
                         self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

########################################################################
