import os
import cv2
import matplotlib.pyplot as plt

class ProcessamentoImagens:
    def __init__(self, pasta_imagens="imagens", tamanho=(128, 128)):
        self.pasta_imagens = pasta_imagens
        self.tamanho = tamanho

    def preprocessar_etapas(self, caminho_imagem):
        img_original = cv2.imread(caminho_imagem)
        img_redimensionada = cv2.resize(img_original, self.tamanho)
        img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
        img_equalizada = cv2.equalizeHist(img_cinza)
        return img_original, img_redimensionada, img_cinza, img_blur, img_equalizada

    def exibir_imagens_processadas(self):
        arquivos = [f for f in os.listdir(self.pasta_imagens) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Total de imagens encontradas: {len(arquivos)}")

        for idx, nome_arquivo in enumerate(arquivos, 1):
            caminho = os.path.join(self.pasta_imagens, nome_arquivo)
            imgs = self.preprocessar_etapas(caminho)

            # Converter para RGB para exibir com matplotlib
            img_original_rgb = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
            img_redimensionada_rgb = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB)

            fig, axs = plt.subplots(1, 5, figsize=(20, 5))
            fig.suptitle(f"{idx}/{len(arquivos)} - {nome_arquivo}", fontsize=16)

            axs[0].imshow(img_original_rgb)
            axs[0].set_title("Original")
            axs[1].imshow(img_redimensionada_rgb)
            axs[1].set_title("Redimensionada (128x128)")
            axs[2].imshow(imgs[2], cmap='gray')
            axs[2].set_title("Cinza")
            axs[3].imshow(imgs[3], cmap='gray')
            axs[3].set_title("Blur Gaussiano")
            axs[4].imshow(imgs[4], cmap='gray')
            axs[4].set_title("Equalização Histograma")

            for ax in axs:
                ax.axis('off')

            plt.show()
