sit_possiveis = ['PNEUMONIA', 'NORMAL']
img_size = 150
def processa_dados(diretorio_dados: str) -> np.array:
    '''
    Recebe um caminho a ser lido : str
    
    Retorna um np.array, contendo um array de imagens e labels
    '''
    raw = [] 
    
    for sit in sit_possiveis: 
        caminho = os.path.join(diretorio_dados, sit)
        classe = sit_possiveis.index(sit)
        for img in os.listdir(caminho):
            try:
                imagem_input = cv2.imread(os.path.join(caminho, img), cv2.IMREAD_GRAYSCALE)
                redimensionada = cv2.resize(imagem_input, (img_size, img_size)) 
                
                raw.append([redimensionada, classe])
                
            except Exception as e:
                print(e)
    
    return np.array(raw)