from FairseqModelLoader import loadModel

if __name__=='__main__':
    model = loadModel(config_path='sample.json')
    print('model', model.parameters())