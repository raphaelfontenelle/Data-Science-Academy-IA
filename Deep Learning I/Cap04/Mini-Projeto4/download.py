########################################################################
#
# Funções para baixar e extrair arquivos de dados da internet.
#
# Implementado no Python 3.7
#
########################################################################

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _print_download_progress(count, block_size, total_size):
    """
    Função utilizada para imprimir o progresso do download.
    Usado como uma função call-back em maybe_download_and_extract ().
    """

    # Percentual concluído
    pct_complete = float(count * block_size) / total_size

    # Mensagem de status. Observe que \r significa que a linha deve sobrescrever a si mesma
    msg = "\r- Progresso do Download: {0:.1%}".format(pct_complete)

    # Print
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):
    """
    Baixa e extrai os dados se ainda não existirem.
    Assume que a url é um arquivo tar-ball.

     : Param url:
         URL da Internet para o arquivo tar para download.
         Exemplo: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

     : Param download_dir:
         Diretório onde o arquivo baixado é salvo.
         Exemplo: "data/CIFAR-10/"

     :Retorna:
         Nada.
    """

    # Nome do arquivo para salvar o arquivo baixado da internet.
    # Usa o nome do arquivo na URL e adiciona ao download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)


    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finalizado. Extraindo os arquivos.")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Concluído.")
    else:
        print("Os dados aparentemente já foram baixados e descompactados.")


########################################################################
