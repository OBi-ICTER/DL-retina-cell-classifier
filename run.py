import argparse
import os 

from classifier.config import Config
import classifier.utils as u
import classifier.model as m


if __name__ == '__main__':

    parser=argparse.ArgumentParser(prog='Image classification')
    parser.add_argument("command")
    parser.add_argument('--image')
    parser.add_argument('--model')
    parser.add_argument('--model_load')
    parser.add_argument('--dataset')
    parser.add_argument('--model_name')
    args = parser.parse_args()

    def img_size(name):
        if name=='seq':
            image_size=Config.SEQ_SIZE
        elif name=='tuning':
            image_size=Config.TL_SIZE
        return image_size

    # Validate arguments
    if args.command == "interfere":
        assert [args.image, args.model, args.model_load, args.dataset]
        path=args.image
        model_path=os.path.join(os.getcwd(), "trainedmodels", args.model_load)
        image_size = img_size(args.model)
        dataset=Config.dataset_evaluate[args.dataset]
        directory=os.path.join(os.getcwd(), dataset)
    elif args.command == "evaluate":
        assert [args.model, args.model_load, args.dataset] 
        model_path=os.path.join(os.getcwd(), "trainedmodels", args.model_load)
        image_size = img_size(args.model)
        dataset=Config.dataset_evaluate[args.dataset]
        directory=os.path.join(os.getcwd(), dataset)
    elif args.command == "train": 
        assert [args.model_name, args.model, args.dataset]
        model_name=os.path.join(os.getcwd(), "trainedmodels", args.model_name)
        image_size = img_size(args.model)
        dataset=Config.dataset_train[args.dataset]
        directory=os.path.join(os.getcwd(), dataset)

    # Train or evaluate
    if args.command == "interfere":
        u.predict(path, model_path, image_size, directory)
    elif args.command == "evaluate":
        x= u.parameters(model_path, image_size, Config.LOOP)
        #u.report(x[0], x[2])
        #u.conf_matrix(x[0], x[1])
        u.roc(x[0], x[1], x[3])
        #u.precision_recall(x[0], x[1], x[3])
    elif args.command == "train":
        if args.model=='seq':
            a,b,c,d = m.data_loader(image_size, directory)
            s=m.sequential(d)
            q,w=m.train_seq(s,a,b,model_name)
            m.plot(q,w)
        elif args.model == 'tuning':
            a,b,c,d = m.data_loader(image_size, directory)
            s=m.sequential(d)
            q,w,e=m.train_seq(s,a,b,model_name)
            m.plot(q,w,e)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
