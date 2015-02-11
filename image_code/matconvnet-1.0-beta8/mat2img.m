function img = mat2img(mat)
img = color(mat);
for i =1:3
   img(:,:,i) = normalize(img(:,:,i)); 
end
end

function normalized = normalize(ch)
normalized = (ch - min(ch(:))) / (max(ch(:)) - min(ch(:)));

end
