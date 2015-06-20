function[batch,Stop]=ReadData(fd_s,parameter)
    tline_s = fgets(fd_s);
    Stop=0;
    doc_index=1;
    sen_index=0;
    i=0;
    max_sen_word_length=0;
    max_doc_sen_length=0;

    indicator=1;
    max_length=0;
    Tag=[];
    while ischar(tline_s)
        text_s=deblank(tline_s);
        if length(text_s)==0
            indicator=1;
            Doc{doc_index}=[i-sen_index+1,i];
            doc_index=doc_index+1;
            if sen_index>max_doc_sen_length
                max_doc_sen_length=sen_index;
            end
            if doc_index==parameter.batch_size+1
                break;
            end
            sen_index=0;
        else
            i=i+1;
            if indicator==1
                tag=str2num(text_s(1));
                Tag=[Tag;tag];
                text_s=text_s(3:length(text_s));
                indicator=0;
            end
            sen_index=sen_index+1;
            Sen{i}=str2num(text_s);
            if length(Sen{i})>max_sen_word_length
                max_sen_word_length=length(Sen{i});
            end
        end
        tline_s = fgets(fd_s);
    end
    if ischar(tline_s)==0
        Stop=1;
    end
    doc_sen_matrix=ones(length(Doc),max_doc_sen_length);
    doc_sen_matrix_r=ones(length(Doc),max_doc_sen_length);
    doc_sen_mask=ones(length(Doc),max_doc_sen_length);
    for i=1:length(Doc)
        L=Doc{i}(1):Doc{i}(2);
        l=length(L);
        doc_sen_matrix(i,max_doc_sen_length-l+1:max_doc_sen_length)=L;
        doc_sen_matrix_r(i,max_doc_sen_length-l+1:max_doc_sen_length)=wrev(L);
        doc_sen_mask(i,1:max_doc_sen_length-l)=0;
    end

    doc_sen_delete={};
    doc_sen_left={};
    for j=1:size(doc_sen_matrix,2)
        doc_sen_delete{j}=find(doc_sen_mask(:,j)==0);
        doc_sen_left{j}=find(doc_sen_mask(:,j)==1);
    end
    sen_word_matrix=ones(length(Sen),max_sen_word_length);
    sen_word_matrix_r=ones(length(Sen),max_sen_word_length);
    sen_word_mask=ones(length(Sen),max_sen_word_length);
    for i=1:length(Sen)
        l=length(Sen{i});
        sen_word_matrix(i,max_sen_word_length-l+1:max_sen_word_length)=wrev(Sen{i});
        sen_word_matrix_r(i,max_sen_word_length-l+1:max_sen_word_length)=Sen{i};
        sen_word_mask(i,1:max_sen_word_length-l)=0;
    end
    sen_word_delete={};
    sen_word_left={};
    for j=1:max_sen_word_length
        sen_word_delete{j}=find(sen_word_mask(:,j)==0);
        sen_word_left{j}=find(sen_word_mask(:,j)==1);
    end
    
    batch.Tag=Tag;
    batch.doc_sen_matrix=doc_sen_matrix;
    batch.doc_sen_matrix_r=doc_sen_matrix_r;
    batch.doc_sen_mask=doc_sen_mask;
    batch.doc_sen_delete=doc_sen_delete;
    batch.doc_sen_left=doc_sen_left;
    batch.sen_word_matrix=sen_word_matrix;
    batch.sen_word_matrix_r=sen_word_matrix_r;
    batch.sen_word_mask=sen_word_mask;
    batch.sen_word_delete=sen_word_delete;
    batch.sen_word_left=sen_word_left;
    
    clear doc_sen_matrix;
    clear doc_sen_mask;
    clear doc_sen_delete;
    clear doc_sen_left;
    clear sen_word_matrix;
    clear sen_word_mask;
    clear sen_word_delete;
    clear sen_word_left;
end
