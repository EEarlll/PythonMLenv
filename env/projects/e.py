# import torchvision
# import torch

# class GLP_model:
#     """GLP_model pretrained using vgg19_bn model"""
#     def __init__(self, device = 'cuda') -> None:
#         # model
#         self.device = device
#         weights = torchvision.models.VGG19_BN_Weights.DEFAULT
#         self.auto_transform = weights.transforms()
#         self.model_5 = torchvision.models.vgg19_bn(weights = weights).to(device)
#         self.class_Names = ['animal giraffe', 'animal lion', 'animal penguin']
#         # freeze layer
#         for i in self.model_5.features.parameters():
#             i.requires_grad = False

#         self.model_5.classifier = torch.nn.Sequential(
#             torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Dropout(p=0.5, inplace=False),
#             torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Dropout(p=0.5, inplace=False),
#             torch.nn.Linear(in_features=4096, out_features=3, bias=True)
#         ).to(device)
        
#         # load state_dict
#         path_rel = r'C:\Users\earle\PythonMLenv\env\projects\Models\02_Giraffe_Lion_Penguin.pt'
#         self.model_5.load_state_dict(torch.load(path_rel))

#     def pred(self, img_path):
#     # predictions 
#         if img_path:
#             target_image = torchvision.io.read_image(str(img_path)).type(torch.float32) / 255
#             # add a batch size (NCHW)
#             target_image = target_image.unsqueeze(0).to(self.device)
#             if target_image.shape[1] != 3:
#                 target_image = target_image[:,:3,:,:]

#             if self.auto_transform:
#                 target_image = self.auto_transform(target_image)
#             self.model_5.to(self.device)
#             self.model_5.eval()
#             with torch.inference_mode():
#                 target_image_logits = self.model_5(target_image)
#                 target_image_probs = torch.softmax(target_image_logits.squeeze(), dim =0)
#                 target_image_pred = torch.argmax(target_image_probs, dim =0).cpu()    

#             return self.class_Names[target_image_pred]




## ----------------------------------------------------------------------
# from datetime import datetime, timedelta
# from tabulate import tabulate
# import pandas as pd


# class Time_conv:
#     def __init__(self) -> None:

#         self.Uutc = datetime.utcnow()
#         self.ref = r"C:\Users\earle\Pythonenv\env\projects\Dataset\Final_output.xlsx"
#         self.df= pd.read_excel(self.ref)
#         self.df['Sovereign state'].fillna(method='ffill', inplace=True)
#         self.df['No. of time zones'].fillna(method='ffill', inplace=True)
#         self.df = self.df[['Sovereign state', 'UTC', 'no.']]
#         self.df_Utclist= self.df['UTC'].tolist()
#         self.df_SovereignStatelist= self.df['Sovereign state'].tolist()


#         print(tabulate(self.df, headers='keys', tablefmt='psql'))
#         while(True):
#             self.country1 = int(input("Enter # of current country: "))
#             self.country2 = int(input("Enter # of country to convert to: "))
#             print()
#             self.conv()
#             print()

#     def conv(self):
#         a = ['+', '±']
#         print(self.df_SovereignStatelist[self.country1] , "  " ,self.df_Utclist[self.country1])
#         Country = self.df_SovereignStatelist[self.country1]
#         Cutc = self.df_Utclist[self.country1]
#         Outc = Cutc[0]
#         utc = Cutc[1:]
#         utc_hours = int(utc[:2])
#         utc_min = int(utc[3:])

#         print(self.df_SovereignStatelist[self.country2] , "  " ,self.df_Utclist[self.country2])
#         Country2 = self.df_SovereignStatelist[self.country2]
#         Cutc2 = self.df_Utclist[self.country2]
#         Outc2 = Cutc2[0]
#         utc2 = Cutc2[1:]
#         utc_hours2 = int(utc2[:2])
#         utc_min2 = int(utc2[3:])

#         if Outc not in a or Outc2 not in a:
#             Outc = 's'
#             Outc2 = 's'

#         if Outc == 's' and Outc2 == 's':
#             self.State = self.Uutc - timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc - timedelta(hours= utc_hours2, minutes=utc_min2)

#         elif Outc == '+' and Outc2 == '+':
#             self.State = self.Uutc + timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc + timedelta(hours= utc_hours2, minutes=utc_min2)
        
#         elif Outc == '+' and Outc2 == '-':
#             self.State = self.Uutc + timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc - timedelta(hours= utc_hours2, minutes=utc_min2)
        
#         elif Outc == 's' and Outc2 == '+':
#             self.State = self.Uutc - timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc + timedelta(hours= utc_hours2, minutes=utc_min2)
        
#         elif Outc == '±' and Outc2 == '+':
#             self.State = self.Uutc
#             self.State2 = self.Uutc + timedelta(hours= utc_hours2, minutes=utc_min2)

#         elif Outc == '±' and Outc2 == 's':
#             self.State = self.Uutc
#             self.State2 = self.Uutc - timedelta(hours= utc_hours2, minutes=utc_min2)
       
#         elif Outc == '+' and Outc2 == '±':
#             self.State = self.Uutc + timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc

#         elif Outc == 's' and Outc2 == '±':
#             self.State = self.Uutc - timedelta(hours= utc_hours, minutes=utc_min)
#             self.State2 = self.Uutc 

#         if self.State > self.State2:
#             Interval = self.State - self.State2
#             print(Country , " is ",self.dhm(Interval), " hours ahead of ", Country2)
#         elif self.State < self.State2:
#             Interval = self.State2 - self.State
#             print(Country2 , " is ",self.dhm(Interval), " hours ahead of ", Country)
#         else:
#             print("There is no time difference between", Country, " and ", Country2)
        
#         print(self.State.strftime("%I:%M %p"), ", in ",Country, " is")
#         print(self.State2.strftime("%I:%M %p"), ", in ", Country2)
        
 
#     def dhm(self,td):
#         return td.seconds//3600


# e = Time_conv()


## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
# from tkinter import *
# import tkinter.ttk as ttk
# from PIL import ImageTk, Image
# from pytube import YouTube
# from threading import *
# from io import BytesIO
# import urllib.request
# import os
# from tkinter import messagebox

# class y:
#     def __init__(self,root) -> None:
#         self.root = root
#         root.title("mp3/mp4 downloader")
#         root.geometry("500x200")
#         root.resizable(False,False)
#         root.iconbitmap(r"C:\Users\earle\Pythonenv\env\projects\elaina.ico")
#         self.videopath = r"C:\Users\earle\Pythonenv\env\projects\Video"
#         self.songpath = r"C:\Users\earle\Pythonenv\env\projects\Song"

#         lefts = Frame(root)
#         leftsbottom = Frame(lefts)
#         rights = Frame(root)
#         bottoms = Frame(root)
#         bottomsright = Frame(bottoms)

#         bottoms.pack(side = BOTTOM , fill=X, )
#         bottomsright.pack(side= RIGHT)
#         rights.pack(side = RIGHT, fill = BOTH)
#         lefts.pack(fill= BOTH, expand = True)
#         leftsbottom.pack(fill= X, side= BOTTOM, expand= True)
    
#         self.label1 = Label(lefts,text ="Channel: ", anchor = "w", justify = LEFT)
#         self.label2 = Label(lefts,text ="Title: ", anchor = "w", justify = LEFT)
#         self.label3 = Label(lefts,text ="Publish Date: ", anchor = "w", justify = LEFT)
#         self.label4 = Label(lefts,text ="Length: ", anchor = "w", justify = LEFT)
#         self.label5 = Label(lefts,text ="Keywords: ", anchor = "w", justify = LEFT)
#         self.progressbar = ttk.Progressbar(leftsbottom, orient=HORIZONTAL, mode = "indeterminate", length = 100)

#         self.image_path= r"C:\Users\earle\Pythonenv\env\projects\placeholderjpg.jpg"
#         self.original = Image.open(self.image_path)
#         resized = self.original.resize((150,150), Image.Resampling.LANCZOS)
#         self.image_ref = ImageTk.PhotoImage(resized)
#         self.image = Label(rights, image = self.image_ref)

#         self.e1 = Entry(bottoms, font =("default", 15))
#         self.b1 = Button(bottomsright, text = "📁", height = 2, width = 11,  command = self.folder)
#         self.b2 = Button(bottomsright, text = "mp3", height = 2, width = 11, command = self.mp3thread)
#         self.b3 = Button(bottomsright, text = "mp4", height = 2, width = 11, command = self.mp4thread)

#         self.image.pack(fill=BOTH, expand=True, padx = 3, pady=5)

#         self.label1.pack(fill=BOTH, expand= True)
#         self.label2.pack(fill=BOTH, expand= True)
#         self.label3.pack(fill=BOTH, expand= True)
#         self.label4.pack(fill=BOTH, expand= True)
#         self.label5.pack(fill=BOTH, expand= True)

#         self.progressbar.pack(fill=X, expand= True,)

        
#         self.e1.pack(fill=BOTH, expand=True, padx = 3, pady=3)
#         self.b1.pack(side = RIGHT, anchor= "se", pady=3, padx= 3)
#         self.b3.pack(side = RIGHT, anchor= "se", pady=3)
#         self.b2.pack(side = RIGHT, anchor= "se" , padx= 3, pady = 3)   

#         self.e1.focus_set()      
    
#     def mp3(self):
#         try:
#             link = self.e1.get()
#             self.progressbar.start()

#             yt = YouTube(link)
#             title = yt.title
#             thumbnail_url = yt.thumbnail_url
#             publishd = yt.publish_date
#             keywords = yt.keywords[:3]
#             author = yt.author
#             length = yt.length
#             self.changeimg(thumbnail_url, title, publishd, keywords, author, length)

#             vid = yt.streams.filter(only_audio=True).first()
#             output = vid.download(self.songpath)
#             base, ext = os.path.splitext(output)
#             mp3_file = base + ".mp3"
#             os.rename(output, mp3_file)

#             self.progressbar.stop()
#         except Exception as e:
#             self.progressbar.stop()
#             messagebox.showerror("Exception occured", e)

#     def mp4(self):
#         try:
#             link = self.e1.get()
#             self.progressbar.start()

#             yt = YouTube(link)
#             title = yt.title
#             thumbnail_url = yt.thumbnail_url
#             publishd = yt.publish_date
#             keywords = yt.keywords[:3]
#             author = yt.author
#             length = yt.length
#             self.changeimg(thumbnail_url, title, publishd, keywords, author, length)

#             video = yt.streams.get_highest_resolution()
#             video.download(self.videopath)
#             self.progressbar.stop()
#         except Exception as e:
#             self.progressbar.stop()
#             messagebox.showerror("Exception occured", e)
            

#     def folder(self):
#         os.startfile(r"C:\Users\earle\Pythonenv\env\projects")

#     def mp3thread(self):
#         t2 = Thread(target = self.mp3)
#         t2.start()

#     def mp4thread(self):
#         t1 = Thread(target= self.mp4)
#         t1.start()


#     def changeimg(self, link, title, publishd, keywords, author, length):
#         self.e1.delete(0, "end")
        
#         self.label1.config(text = "Channel: " + author)
#         self.label2.config(text = "Title: " + title)
#         self.label3.config(text = "Publish date: " + str(publishd))
#         self.label4.config(text = "Length : " + str(length) + " seconds")
#         self.label5.config(text = "Keywords: " + ", ".join(keywords))
    
#         u = urllib.request.urlopen(link)
#         raw_data = u.read()
#         u.close()

#         self.orig = Image.open(BytesIO(raw_data))
#         resized = self.orig.resize((150,150), Image.Resampling.LANCZOS)
#         self.im_ref = ImageTk.PhotoImage(resized)

#         self.image.configure(image=self.im_ref)

#     def exceptions(self):
#         pass
    

# root = Tk()
# gui = y(root)
# root.mainloop()
#-----------------------------------------------------------------------

