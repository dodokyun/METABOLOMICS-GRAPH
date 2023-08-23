source("/home/n1/dokyunkim/UKBB_METAB_FINAL/icd10disease_230504.R")
library(data.table)

Dementia <-c("F00", "F01", "F02", "F03", "G30", "G31")
MACE <-c("G45", "I21", "I22", "I23", "I24", "I25", "I63", "I64")
T2_Diabetes <-c("E10", "E11", "E12", "E13", "E14")
Liver_Disease <-c("B15", "B16", "B17", "B18", "B19", "C22", "E83", "E88", "I85", "K70", "K72", "K73", "K74", "K75", "K76", "R18", "Z94")
Renal_Disease <-c('N00', 'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N09', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N25', 'N26', 'N27', 'N28', 'N29')
Atrial_Fibrillation <-c("I48")
Heart_Failure <-c("I50")
CHD <-c('I20', 'I21', 'I22', 'I23', 'I24', 'I25')
Venous_Thrombosis <-c("I80", "I81", "I82")
Cerebral_Stroke <-c("I63", "I65", "I66")
AAA <-c("I71")
PAD <-c('I70', 'I71', 'I72', 'I73', 'I74', 'I75', 'I76', 'I77', 'I78', 'I79')
Asthma <-c("J45", "J46")
COPD <-c("J40", "J41", "J42", "J43", "J44", "J47")
Lung_Cancer <-c("C33", "C34")
Skin_Cancer <-c("C44")
Colon_Cancer <-c("C18")
Rectal_Cancer<-c("C19", "C20")
Prostate_Cancer <-c("C61")
Breast_Cancer <-c("C50")
Parkinson <-c("G20", "G21", "G22")
Fractures <-c("S02", "S12", "S22", "S32", "S42", "S52", "S62", "S72", "S82", "S92", "T02", "T08", "T10")
Cataracts <-c("H25", "H26")
Glaucoma <-c("H40")

dlist <- list(Dementia,MACE,T2_Diabetes,Liver_Disease,Renal_Disease,Atrial_Fibrillation,Heart_Failure,CHD,Venous_Thrombosis,
	Cerebral_Stroke,AAA,PAD,Asthma,COPD,Lung_Cancer,Skin_Cancer,Colon_Cancer,Rectal_Cancer,Prostate_Cancer,Breast_Cancer,Parkinson,
	Fractures,Cataracts,Glaucoma)

dlist2 <- list('Dementia','MACE','T2_Diabetes','Liver_Disease','Renal_Disease','Atrial_Fibrillation','Heart_Failure','CHD','Venous_Thrombosis',
	'Cerebral_Stroke','AAA','PAD','Asthma','COPD','Lung_Cancer','Skin_Cancer','Colon_Cancer','Rectal_Cancer','Prostate_Cancer','Breast_Cancer','Parkinson',
	'Fractures','Cataracts','Glaucoma')

# ammend outcome data frame into one list 
outcome <- list()
for (i in 1:length(dlist)){
	outcome <- append(outcome,list(icd10_disease(dlist[[i]])))

}

# save data frame separately 
for (i in 1:length(dlist2)){
	disease_name <- dlist2[i]
	filename <- paste0("/home/n1/dokyunkim/UKBB_METAB_FINAL/Data/multi_disease/",disease_name,".csv")
	write.csv(outcome[i],file=filename, quote=FALSE, row.names=FALSE)

}

load(file= "/home/n1/dokyunkim/UKBB_METAB_FINAL/Data/multi_disease/outcome.RData")


for (i in 1:24){

	print(sum((outcome[[i]]$prevalence == 0) & (outcome[[i]]$incidence ==1)) /  dim(outcome[[i]])[1] * 100)
}
sum((outcome[[1]]$prevalence == 0) & (outcome[[1]]$incidence ==1)) /  dim(outcome[[1]])[1] * 100





#  1: death -> 3 (in status) / 2,3,4,5: left or lost(in loss file) -> 2 (in status) / Nothing (alive) -> 0 (in status) / disease -> 1 (later in use of)  
censor <- fread( "/media/leelabsg-storage0/UKBB_WORK/METABOLOMICS_WORK/censoring_and_losttofollowup_and_death_earliest_date.csv")

dd <- data.table(outcome[[1]])
censor$f.eid <- as.character(censor$f.eid)


# merge diseaes data and censoring data 
setkey(censor, f.eid)
setkey(dd, f.eid)
dd2 <- censor[dd]


# create status & tte column using below logic 
#  1: death -> 3 (in status) / 2,3,4,5: left or lost(in loss file) -> 2 (in status) / Nothing (alive) -> 0 (in status) /
#     disease -> 1 (later in use of) / prevalence =1 -> NA 
    '
    p   0   0   1   1
    i   0   1   0   1
        c   e   0   t

        p : prevalence 
        i : incidence
        c : censoring date 
        e : earlier one
        t : tte 
    '
dd2$status1 <- ifelse(dd2$prevalence ==0, ifelse(dd2$incidence ==0, dd2$status,     ifelse(dd2$censor_tte < dd2$tte, dd2$status, dd2$incidence)),  NA)# previously NA was 1,be careful
dd2$tte1    <- ifelse(dd2$prevalence ==0, ifelse(dd2$incidence ==0, dd2$censor_tte, ifelse(dd2$censor_tte < dd2$tte, dd2$censor_tte, dd2$tte)), NA)    

# remove rows when blood_date is missing 
dd3 <- dd2[!is.na(dd2$blood_date),]

# 
dd4 <- dd3[,c('f.eid','status1','tte1')]



#  1: death -> 3 (in status) / 2,3,4,5: left or lost(in loss file) -> 2 (in status) / Nothing (alive) -> 0 (in status) / disease -> 1 (later in use of)  
    '
    p   0   0   1   1
    i   0   1   0   1
        c   e   0   t

        p : prevalence 
        i : incidence
        c : censoring date 
        e : earlier one
        t : tte 
    '
