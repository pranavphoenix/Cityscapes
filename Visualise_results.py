test_class = MyClass('/content/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)
test_loader=torch.utils.data.DataLoader(test_class, batch_size=1, 
                      shuffle=True)
mIoU = 0
model.eval()
with torch.no_grad():
    for batch in test_loader:
        img,seg=batch
        output=model(img.cuda())
        segment=encode_segmap(seg.cuda())
        mIoU = iou_pytorch(output, segment)
        print(mIoU)
        if mIoU > 0.99:
          break
print(img.shape,seg.shape,output.shape) 

from torchvision import transforms
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)


sample=0
invimg=inv_normalize(img[sample])
outputx=output.detach().cpu()[sample]
encoded_mask=encode_segmap(seg[sample].clone()) 
decoded_mask=decode_segmap(encoded_mask.clone())  
decoded_ouput=decode_segmap(torch.argmax(outputx,0))
fig,ax=plt.subplots(ncols=3,figsize=(25,50),facecolor='white')  
ax[0].imshow(np.moveaxis(invimg.numpy(),0,2)) 

ax[1].imshow(decoded_mask) 
ax[2].imshow(decoded_ouput) 
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].set_title('Input Image')
ax[1].set_title('Ground mask')
ax[2].set_title('Predicted mask')
plt.savefig('result.png',bbox_inches='tight')
