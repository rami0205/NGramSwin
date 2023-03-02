dataset_list = ["Set5", "Set14", "BSDS100", "urban100", "manga109"];
dl = size(dataset_list);
dl = dl(2);
for scale=2:4
    for d=1:dl
        dataset=dataset_list(d);
        file_list = dir(dataset+"/HR");
        end_idx = size(file_list)-2;
        for i=1:end_idx
            file = file_list(i+2).name;
            file = split(file, ".");
            file = file(1);
            I = imread(dataset+"/HR/"+file+".png");
            img_size = size(I);
            h = img_size(1);
            w = img_size(2);
            J = imresize(I, 1/scale, "bicubic");
            J = J(1:fix(h/scale),1:fix(w/scale),:);
            save_path = dataset+"/LR_bicubic"+"/X"+scale+"/"+file+"x"+scale+".png";
            imwrite(J, save_path);
        end
    end
end
