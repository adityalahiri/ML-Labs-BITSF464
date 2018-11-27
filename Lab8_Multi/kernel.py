---
title: "LDA, PCA, & Boosting with Engineered Features"
author: "Mikhail Lara"
output: 
    html_document:
        code_folding: hide
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
options(warn = -1)
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))
suppressMessages(library(e1071))
suppressMessages(library(MASS))
suppressMessages(library(caret))
suppressMessages(library(gbm))

filename<-'../input/glass.csv'
DT<-fread(filename)
```

The data consists of 214 observations of 6 different types of glass. The number of observations in the dataset vary significantly across the glass types, which could pose a problem if ML techniques are used without data preprocessing. 
```{r}
invisible(DT[,Type:=as.factor(Type)])
print('Number of Occurrences for Each Glass Type')
print(DT[,.N,by=Type])
```

###Brute Force Stochastic Gradient Boosting without Pre-Processing

A stochastic gradient boosting model is fit to the raw data without pre-processing. The out-of-sample accuracy is estimated using leave-one-out cross-validation(CV). 

The final model consists of **150 trees** with **interaction depth of 2**. 

The CV **Accuracy** is **0.7850467**.

The CV **Concordance(kappa)** is **0.7028226**.

```{r}
DT.brute<-data.table(DT)
invisible(DT.brute[,Type:=as.factor(as.character(Type))])

train_controlA<- trainControl(method="LOOCV")

set.seed(123)

suppressMessages(gbm.fit.brute1<- train(Type~., data=DT.brute, trControl=train_controlA, method="gbm",verbose=FALSE))

gbm.fit.brute1
#confusionMatrix(predict(gbm.fit.brute1,DT.brute),DT.brute$Type)
remove(DT.brute)
```
##Exploratory Analysis & Modeling

**Discrimination Criterion:** A unique characteristic of type 6 glass is that none of the recorded observations have any levels of potassium(K), iron(Fe), or barium(Ba). The absense of these three elements in the glass can be used to uniquely distinguish type 6 glass from the other 5 types. This helps with modeling because the glass type with the fewest occurrences can be omitted.

The **Accuracy** is **0.9766355**.

The **Sensitivity(recall)** is **1**.

The **True Negative Rate(TNR)** is **0.9756098**.

The **Positive Predictive Value(PPV)** is **0.6428571**.

The **Negative Predictive Value(NPV)** is **1**.


```{r }
print('Summary of Type 6 Glass(Tableware)')
print(summary(DT[Type=='6']))

#Cleaning
#Processing: OMIT ANY Type 6 Glass (Minor Cross-Over with Groups 1[1],2[3],3[1])
accuracy<-(9+(nrow(DT)-nrow(DT[((K==0)&(Fe==0)&(Ba==0))])))/nrow(DT)
recall<-9/nrow(DT[Type=='6'])
TNR<-(nrow(DT)-nrow(DT[((K==0)&(Fe==0)&(Ba==0))]))/(nrow(DT[Type!='6'])) #specificity
PPV<-nrow(DT[Type=='6'])/nrow(DT[((K==0)&(Fe==0)&(Ba==0))])
NPV<-(nrow(DT)-nrow(DT[((K==0)&(Fe==0)&(Ba==0))]))   /   ((nrow(DT)-nrow(DT[((K==0)&(Fe==0)&(Ba==0))])))
```

###Observations with Zero Potassium, Iron, and Barium
```{r}
DT[((K==0)&(Fe==0)&(Ba==0)),.N,by=Type]
DT.omit6 <- DT[!((K==0)&(Fe==0)&(Ba==0))]
```

###Distribution of Remaining Glass Types Data
```{r}
DT.omit6[,.N,by=Type]
```

###Pre-Processing: Outlier Identification

Outliers of the in-group attribute distributions needs to be removed before doing feature engineering. This step is particularly important for principle component analysis(PCA) since that algorithm is highly sensitive to spurrious data.

Since the in-group attribute distributions are **not normally distributed**,  outliers are identified as points that are 'significantly far' outside of the 25% - 75% quantile limits.

```{r}
g_RI<-ggplot(data=DT.omit6,aes(y=RI,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+ geom_jitter(width=.1)

g_Na<-ggplot(data=DT.omit6,aes(y=Na,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+ 
    geom_jitter(width=.1) # Group 7 has higher Na than other groups & is Normal (p=0.001)

g_Mg<-ggplot(data=DT.omit6,aes(y=Mg,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+
    geom_jitter(width=.1) # Groups 1,2,3 have much higher Mg than 5&7

g_Al<-ggplot(data=DT.omit6,aes(y=Al,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+
    geom_jitter(width=.1) # Groups 5 & 7 have higher and distinct Al levels than other groups

g_Si<-ggplot(data=DT.omit6,aes(y=Si,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+ geom_jitter(width=.1)

g_Ca<-ggplot(data=DT.omit6,aes(y=Ca,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+ geom_jitter(width=.1)  

g_Fe<-ggplot(data=DT.omit6,aes(y=Fe,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+
    geom_jitter(width=.1) # Group 7 has lower Fe than other groups 

grid.arrange(g_RI,g_Na,g_Mg,ncol=3,
             top='Elements without Clear, Discernable Outliers (1)')
grid.arrange(g_Al, g_Si,g_Ca,g_Fe,ncol=3,
             top='Elements without Clear, Discernable Outliers (2)')
```


The glass attributes that have 'clear, discernable' outliers according to the quantile criterion are Ba and K. A total of **6 outliers** are removed based on this criterion.

```{r}
g_Ba<-ggplot(data=DT.omit6,aes(y=Ba,x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+
    geom_jitter(width=.1) # For Group 5 & 7, 7 is somewhat normal (p=0.001) 

g_K<-ggplot(data=DT.omit6,aes(y=K, x=as.factor(Type),fill=as.factor(Type)))+
    xlab('Type')+scale_fill_discrete(guide=FALSE)+
    geom_violin(draw_quantiles=c(0.25,0.75))+
    geom_jitter(width=.1) # Groups 7 skew higher for K than other groups

grid.arrange(g_Ba,g_K,ncol=3, top='Elements with Clear, Discernable Outliers')
```

```{r}
#Remove Barium Outliers
#(Ba>2)
#(Type=='1')&(Ba>0.5)

#Remove K Outliers
type5.avg.K<-mean(DT.omit6[Type=='5']$K)
type5.sd.K<-sd(DT.omit6[Type=='5']$K)
thresh.type5.sd.K<-type5.avg.K+2*sd(DT.omit6[Type=='5']$K)
#(K>thresh.type5.sd.K)&(Type==5)

DT.clean<-DT.omit6[!((Ba>2)|(Type=='1')&(Ba>0.5)|(K>thresh.type5.sd.K)&(Type==5) )]
```

###Glass Type Distribution with Outliers Removed
```{r}
DT.clean[,.N,by=Type]
```

##Linear Discriminant Analysis
```{r}
lda.fit<-lda(Type~.,data = DT.clean)
lda.fit

DT.lda<-as.matrix(DT.clean[,-c('Type'),with=FALSE])
DT.lda<- DT.lda %*% lda.fit$scaling
DF.lda<-data.frame(DT.lda,DT.clean$Type)
names(DF.lda)<-c( "LD1"     ,      "LD2"     ,      "LD3"     ,      "LD4"         ,  "Type")

ggplot(data=DF.lda, aes(x=LD1,y=LD2,colour=as.factor(Type)))+geom_point()

glda1<-ggplot(data=DF.lda, aes(x=LD1,fill=as.factor(Type)))+geom_density(alpha=0.25) # Type 5 & 7
glda2<-ggplot(data=DF.lda, aes(x=LD2,fill=as.factor(Type)))+geom_density(alpha=0.25) # Type 5
glda3<-ggplot(data=DF.lda, aes(x=LD3,fill=as.factor(Type)))+geom_density(alpha=0.25) 
glda4<-ggplot(data=DF.lda, aes(x=LD4,fill=as.factor(Type)))+geom_density(alpha=0.25) # Type 3?

grid.arrange(glda1,glda2,glda3,glda4,ncol=2,top='Class Separation by Linear Discriminants')
```

##Principle Component Analysis
```{r }
PCA<-prcomp(x=DT.clean[,-c('Type'),with=FALSE])
PCA

DT.pca<-as.matrix(DT.clean[,-c('Type'),with=FALSE])
DT.pca<- DT.pca %*% PCA$rotation
DF.pca<-data.frame(DT.pca,DT.clean$Type)

gpc1<-ggplot(data=DF.pca, aes(x=PC1,fill=as.factor(DF.pca[,10])))+geom_density(alpha=0.25) 
gpc2<-ggplot(data=DF.pca, aes(x=PC2,fill=as.factor(DF.pca[,10])))+geom_density(alpha=0.25)
gpc3<-ggplot(data=DF.pca, aes(x=PC3,fill=as.factor(DF.pca[,10])))+geom_density(alpha=0.25) 
gpc9<-ggplot(data=DF.pca, aes(x=PC9,fill=as.factor(DF.pca[,10])))+geom_density(alpha=0.25) 

grid.arrange(gpc1,gpc2,gpc3,gpc9,ncol=2,top='Class Separation by Principle Components')
```

##Feature Engineering for (123), 5, & 7

* F1
    + PC2 Separates Type 7 from Rest 
* F2
    + Ba 25% Quantile, is higher that all other class maxima [Higher Ba ~ Higher Prob(Type 7)] 
    + Can use Non-Parametric Bootstrap Simulations to Determine Quantile Threshold
* F3
    + PC3 Mildly Separates Type 5 from Rest 
* F4
    + LD2 clearly separates Type 5
* F5
    + PC9 Mildly Separates Type 3 from Rest
* F6
    + Has Zero Potassium, Barium, or Iron (Unique Characteristic for that type of glass)
* F7
    + PC1 Separates 5 & 7 from Rest
* F8
    + LD1 May Be Used To Identity Groups 5 & 7 
    + Making the Distributions More Peaked (kurtosis) May Help
* F9
    + Mean Mg of Combined Groups are Less than Mean Mg of Combined Groups 1,2,3 
    + Use Non-Parametric Bootstrap Simulations to Determine Threshold 

```{r }
DT.feat<-data.table(DT.clean)

pca.2<-PCA$rotation[,2]
invisible(DT.feat[,F1:=RI*pca.2[1] +Na*pca.2[2] +Mg*pca.2[3] +Al*pca.2[4] +Si*pca.2[5] +K*pca.2[6] +Ca*pca.2[7] +Ba*pca.2[8] +Fe*pca.2[9]])

sims<-matrix(nrow=10000,ncol=50)
quant<-vector(mode='numeric',length=10000)
samps.sin7<- DT.clean[(Type!='7')]$Ba
set.seed(123)
for(i in 1:10000){
    sims[i,]<-sample(x=samps.sin7, size=50,replace=TRUE)
    quant[i]<-quantile(sims[i,],probs=0.95)
}
Ba.Thresh<-mean(quant)
Ba.type7.mean<-mean(DT.feat[Type=='7']$Ba)
invisible(DT.feat[,F2:=ifelse(Ba>Ba.Thresh,exp(3.25/(abs(Ba-Ba.type7.mean)^1.7+Ba.type7.mean)),Ba)])

pca.3<-PCA$rotation[,3]
invisible(DT.feat[,F3:=RI*pca.3[1] +Na*pca.3[2] +Mg*pca.3[3] +Al*pca.3[4] +Si*pca.3[5] +K*pca.3[6] +Ca*pca.3[7] +Ba*pca.3[8] +Fe*pca.3[9]])

ld.2<-lda.fit$scaling[,2]
invisible(DT.feat[,F4:=RI*ld.2[1] +Na*ld.2[2] +Mg*ld.2[3] +Al*ld.2[4] +Si*ld.2[5] +K*ld.2[6] +Ca*ld.2[7] +Ba*ld.2[8] +Fe*ld.2[9]])

pca.9<-PCA$rotation[,9]
invisible(DT.feat[,F5:=RI*pca.9[1] +Na*pca.9[2] +Mg*pca.9[3] +Al*pca.9[4] +Si*pca.9[5] +K*pca.9[6] +Ca*pca.9[7] +Ba*pca.9[8] +Fe*pca.9[9]])

pca.1<-PCA$rotation[,1]
invisible(DT.feat[,F7:=RI*pca.1[1] +Na*pca.1[2] +Mg*pca.1[3] +Al*pca.1[4] +Si*pca.1[5] +K*pca.1[6] +Ca*pca.1[7] +Ba*pca.1[8] +Fe*pca.1[9]])

ld.1<-lda.fit$scaling[,1]
invisible(DT.feat[,F8:=RI*ld.1[1] +Na*ld.1[2] +Mg*ld.1[3] +Al*ld.1[4] +Si*ld.1[5] +K*ld.1[6] +Ca*ld.1[7] +Ba*ld.1[8] +Fe*ld.1[9]])

sims<-matrix(nrow=10000,ncol=50)
quant<-vector(mode='numeric',length=10000)
samps<- DT.clean[(Type!='7')&(Type!='5')]$Mg
for(i in 1:10000){
    sims[i,]<-sample(samps, size=50,replace=TRUE)
    quant[i]<-quantile(sims[i,],probs=0.25)
}
F9_Thresh<-mean(quant)
invisible(DT.feat[,F9:=ifelse(Mg>=F9_Thresh,(Mg-F9_Thresh)^2,0)])

invisible(DT.feat[,group:=ifelse(Type=='7','7',ifelse(Type=='5','5','123'))])

invisible(DT.feat[,RI:=NULL])
invisible(DT.feat[,Na:=NULL])
invisible(DT.feat[,Mg:=NULL])
invisible(DT.feat[,Al:=NULL])
invisible(DT.feat[,Si:=NULL])
invisible(DT.feat[,K:=NULL])
invisible(DT.feat[,Ca:=NULL])
invisible(DT.feat[,Ba:=NULL])
invisible(DT.feat[,Fe:=NULL])

```

##Linear Discriminant Analysis with Engineered Features
```{r}
lda.feat<-lda(group~.,data = DT.feat[,-c('Type'),with=FALSE])
lda.feat
#lda.feat$svd

mat.lda.feat<-as.matrix(DT.feat[,-c('group','Type'),with=FALSE])
mat.lda.feat<- mat.lda.feat %*% lda.feat$scaling
DF.lda.feat<-data.frame(mat.lda.feat,DT.feat$group)
names(DF.lda.feat)<-c('LD1','LD2','group')

g11<-ggplot(data=DF.lda.feat, aes(x=LD1,fill=as.factor(group)))+geom_density(alpha=0.25) 
g12<-ggplot(data=DF.lda.feat, aes(x=LD2,fill=as.factor(group)))+geom_density(alpha=0.25) 

grid.arrange(g11,g12,ncol=2)

ggplot(data=DF.lda.feat, aes(x=LD1,y=LD2,colour=as.factor(group)))+geom_point()+ggtitle('Separation by Engineered Linear Discriminants')
```

##Principle Component Analysis with Engineered Features
```{r}
PCA.feat<-prcomp(x=DT.feat[,-c('group','Type'),with=FALSE])
summary(PCA.feat)

mat.pca.feat<-as.matrix(DT.feat[,-c('group','Type'),with=FALSE])
mat.pca.feat<- mat.pca.feat %*% PCA.feat$rotation
DF.pca.feat<-data.frame(mat.pca.feat,DT.feat$group)

g21<-ggplot(data=DF.pca.feat, aes(x=PC1,fill=as.factor(DT.feat.group)))+geom_density(alpha=0.25) 
g22<-ggplot(data=DF.pca.feat, aes(x=PC2,fill=as.factor(DT.feat.group)))+geom_density(alpha=0.25) 

grid.arrange(g21,g22,ncol=2)
```

##Gradient Boosting for Types (123),5,and 7

PCA & LDA with the engineered features can distinguish between the glass types using the **2 linear discriminants** and the **first 2 principal components**. 

* A single stochastic gradient boosting model can be built to classify the glass as:
    + Type 7 Glass
+ Type 5 Glass
+ Type 1, 2, or 3 Glass (Will Use a Second Model to Sub-Classify)

The final model consists of **150 trees** with **interaction depth of 1**. 
The CV **accuracy** is **0.9330499**.
The CV **concordance(kappa)** is **0.8038155**.

```{r}
DT.gbm<-data.table(DF.lda.feat,DF.pca.feat$PC1,DF.pca.feat$PC2)
setnames(DT.gbm,old=names(DT.gbm),new=c('LD1','LD2','group','PC1','PC2'))

# define training control
train_controlA<- trainControl(method="LOOCV")
train_controlB<- trainControl(method="cv", number=(5))

# train the model 
set.seed(123)
gbm.fit<- train(group~., data=DT.gbm, trControl=train_controlB, method="gbm",verbose=FALSE)

gbm.fit
plot(gbm.fit)
#summary(gbm.fit)
#confusionMatrix(predict(gbm.fit,DT.gbm),DT.gbm$group)

```

## Gradient Boosting for Types 1, 2, & 3

* A second stochastic gradient boosting model can be built to classify the glass as:
    + Type 1, 2, or 3 Glass (Will Use a Second Model to Sub-Classify)

The final model consists of **100 trees** with **interaction depth of 3**. 
The CV **accuracy** is **0.8141026*.
The CV **concordance(kappa)** is **0.6771339**.

```{r}
DT123.clean<-data.table(DT.clean[(Type=='1')|(Type=='2')|(Type=='3')])
setattr(DT123.clean$Type,"levels",c('1','2','3'))

# define training control
train_controlA<- trainControl(method="LOOCV")

# train the model 
set.seed(123)
gbm.fit123<- train(Type~.-Ba, data=DT123.clean, trControl=train_controlA, method="gbm",verbose=FALSE)

gbm.fit123
plot(gbm.fit123)
#summary(gbm.fit123)
#confusionMatrix(predict(gbm.fit123,DT123.clean),DT123.clean$Type)

```

#Summary

* Brute Force Gradient Boosting (Control)
    + CV **Accuracy** is **0.7850467**.
    + CV **Concordance(kappa)** is **0.7028226**.

* Omit 6
    + **Accuracy** is **0.9766355**.
    + **Sensitivity(recall)** is **1**.
    + **True Negative Rate(TNR)** is **0.9756098**.
    + **Positive predictive value(PPV)** is **0.6428571**.
    + **Negative predictive value(NPV)** is **1**.
    
* Gradient Boosting for Types (123), 5, & 7
    + CV **Accuracy** is **0.9330499**.
    + CV **Concordance(kappa)** is **0.8038155**.

* Gradient Boosting for Types 1,2, and 3
    + CV **Accuracy** is **0.8141026**.
    + CV **Concordance(kappa)** is **0.6771339**.

