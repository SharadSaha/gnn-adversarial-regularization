??&
?%?$
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??#
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
GraphRegularization/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameGraphRegularization/total

-GraphRegularization/total/Read/ReadVariableOpReadVariableOpGraphRegularization/total*
_output_shapes
: *
dtype0
?
GraphRegularization/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameGraphRegularization/count

-GraphRegularization/count/Read/ReadVariableOpReadVariableOpGraphRegularization/count*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?I
value?IB?I B?I
?

base_model
nbr_features_layer
regularizer
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?

layer_with_weights-0

layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
layer_with_weights-4
layer-8
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
 
h

"kernel
#bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

$kernel
%bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
R
=	variables
>trainable_variables
?regularization_losses
@	keras_api
h

&kernel
'bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
h

(kernel
)bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
h

*kernel
+bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
 
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
F
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

_0
`1
a2
 

ascaled_graph_loss

"0
#1

"0
#1
 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
1	variables
2trainable_variables
3regularization_losses
 
 
 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
5	variables
6trainable_variables
7regularization_losses

$0
%1

$0
%1
 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
9	variables
:trainable_variables
;regularization_losses
 
 
 
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
=	variables
>trainable_variables
?regularization_losses

&0
'1

&0
'1
 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses

(0
)1

(0
)1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses

*0
+1

*0
+1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
 
?

0
1
2
3
4
5
6
7
8

?0
?1
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
ca
VARIABLE_VALUEGraphRegularization/total4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEGraphRegularization/count4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
\Z
VARIABLE_VALUEtotal_2?base_model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEcount_2?base_model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
\Z
VARIABLE_VALUEtotal_3?base_model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEcount_3?base_model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_NL_nbr_0_imagePlaceholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
serving_default_NL_nbr_0_weightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_NL_nbr_1_imagePlaceholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
serving_default_NL_nbr_1_weightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_NL_nbr_2_imagePlaceholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
serving_default_NL_nbr_2_weightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_imagePlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_NL_nbr_0_imageserving_default_NL_nbr_0_weightserving_default_NL_nbr_1_imageserving_default_NL_nbr_1_weightserving_default_NL_nbr_2_imageserving_default_NL_nbr_2_weightserving_default_imageconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasGraphRegularization/totalGraphRegularization/count*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6984
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-GraphRegularization/total/Read/ReadVariableOp-GraphRegularization/count/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_8615
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcounttotal_1count_1GraphRegularization/totalGraphRegularization/counttotal_2count_2total_3count_3Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_8760??"
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5703

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?3
?
yGraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_5489?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shape
{graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
|
xgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shape?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
xGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
xgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
Amean_squared_error_assert_broadcastable_is_valid_shape_false_6272F
Bmean_squared_error_assert_broadcastable_is_valid_shape_placeholder
?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_values_rank?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_weights_rank?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_values_shape?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_weights_shapeC
?mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_values_rank?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
Pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfamean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_values_shape?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_weights_shapeamean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *n
else_branch_R]
[mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_6281*
output_shapes
: *m
then_branch^R\
Zmean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_6280?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
?mean_squared_error_assert_broadcastable_is_valid_shape_identityHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8355

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
fgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7212k
ggraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderm
igraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
h
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
=mean_squared_error_assert_broadcastable_AssertGuard_true_6325?
|mean_squared_error_assert_broadcastable_assertguard_identity_mean_squared_error_assert_broadcastable_is_valid_shape_identity
C
?mean_squared_error_assert_broadcastable_assertguard_placeholderE
Amean_squared_error_assert_broadcastable_assertguard_placeholder_1E
Amean_squared_error_assert_broadcastable_assertguard_placeholder_2
B
>mean_squared_error_assert_broadcastable_assertguard_identity_1
V
8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity|mean_squared_error_assert_broadcastable_assertguard_identity_mean_squared_error_assert_broadcastable_is_valid_shape_identity9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
>mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityEmean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
>mean_squared_error_assert_broadcastable_assertguard_identity_1Gmean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
[mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_8241?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
a
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholderc
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_1c
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_2
`
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
t
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identitycmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1emean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
\GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_5534?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
b
^graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholderd
`graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_1d
`graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_2
a
]graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
u
WGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityX^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
]GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentitydGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
]graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1fGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
)__inference_sequential_layer_call_fn_7804

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_1_layer_call_fn_8345

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5703?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8380

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????KK i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????KK w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????KK
 
_user_specified_nameinputs
?
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7306?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
^mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_6365?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
d
`mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholderf
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_1f
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_2f
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_3a
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?

?
)__inference_sequential_layer_call_fn_5860	
image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5837o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameimage
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8360

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_8760
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: :
 assignvariableop_5_conv2d_kernel:,
assignvariableop_6_conv2d_bias:<
"assignvariableop_7_conv2d_1_kernel:.
 assignvariableop_8_conv2d_1_bias:<
"assignvariableop_9_conv2d_2_kernel: /
!assignvariableop_10_conv2d_2_bias: 4
 assignvariableop_11_dense_kernel:
??@,
assignvariableop_12_dense_bias:@4
"assignvariableop_13_dense_1_kernel:@.
 assignvariableop_14_dense_1_bias:#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: 7
-assignvariableop_19_graphregularization_total: 7
-assignvariableop_20_graphregularization_count: %
assignvariableop_21_total_2: %
assignvariableop_22_count_2: %
assignvariableop_23_total_3: %
assignvariableop_24_count_3: B
(assignvariableop_25_adam_conv2d_kernel_m:4
&assignvariableop_26_adam_conv2d_bias_m:D
*assignvariableop_27_adam_conv2d_1_kernel_m:6
(assignvariableop_28_adam_conv2d_1_bias_m:D
*assignvariableop_29_adam_conv2d_2_kernel_m: 6
(assignvariableop_30_adam_conv2d_2_bias_m: ;
'assignvariableop_31_adam_dense_kernel_m:
??@3
%assignvariableop_32_adam_dense_bias_m:@;
)assignvariableop_33_adam_dense_1_kernel_m:@5
'assignvariableop_34_adam_dense_1_bias_m:B
(assignvariableop_35_adam_conv2d_kernel_v:4
&assignvariableop_36_adam_conv2d_bias_v:D
*assignvariableop_37_adam_conv2d_1_kernel_v:6
(assignvariableop_38_adam_conv2d_1_bias_v:D
*assignvariableop_39_adam_conv2d_2_kernel_v: 6
(assignvariableop_40_adam_conv2d_2_bias_v: ;
'assignvariableop_41_adam_dense_kernel_v:
??@3
%assignvariableop_42_adam_dense_bias_v:@;
)assignvariableop_43_adam_dense_1_kernel_v:@5
'assignvariableop_44_adam_dense_1_bias_v:
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_dense_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_graphregularization_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_graphregularization_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_3Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv2d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
_GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_5480?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
e
agraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholderg
cgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_1g
cgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_2g
cgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_3b
^graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
^graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitygGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
'__inference_conv2d_1_layer_call_fn_8329

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_conv2d_layer_call_fn_8289

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5736y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?V
?
__inference__traced_save_8615
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_graphregularization_total_read_readvariableop8
4savev2_graphregularization_count_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB?base_model/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_graphregularization_total_read_readvariableop4savev2_graphregularization_count_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::::: : :
??@:@:@:: : : : : : : : : : ::::: : :
??@:@:@:::::: : :
??@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :& "
 
_output_shapes
:
??@: !

_output_shapes
:@:$" 

_output_shapes

:@: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :&*"
 
_output_shapes
:
??@: +

_output_shapes
:@:$, 

_output_shapes

:@: -

_output_shapes
::.

_output_shapes
: 
?
J
.__inference_max_pooling2d_2_layer_call_fn_8390

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&& * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????&& "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????KK :W S
/
_output_shapes
:?????????KK 
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8315

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?;
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6727

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6)
sequential_6677:
sequential_6679:)
sequential_6681:
sequential_6683:)
sequential_6685: 
sequential_6687: #
sequential_6689:
??@
sequential_6691:@!
sequential_6693:@
sequential_6695:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?"graph_loss/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1=
ShapeShapeinputs_6*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPackinputsinputs_2inputs_4*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:x
stack_1Packinputs_1inputs_3inputs_5*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:??????????
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputs_6sequential_6677sequential_6679sequential_6681sequential_6683sequential_6685sequential_6687sequential_6689sequential_6691sequential_6693sequential_6695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6585?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallReshape:output:0sequential_6677sequential_6679sequential_6681sequential_6683sequential_6685sequential_6687sequential_6689sequential_6691sequential_6693sequential_6695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6585?
"graph_loss/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
mulMulmul/x:output:0+graph_loss/StatefulPartitionedCall:output:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: |

Identity_1Identity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1#^graph_loss/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12H
"graph_loss/StatefulPartitionedCall"graph_loss/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????&& "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????KK :W S
/
_output_shapes
:?????????KK 
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5715

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7655?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?)
?
]GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_5535?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
a
]graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
??YGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*0
value'B% BGraphRegularization/Reshape_1:0?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*V
valueMBK BEGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:0?
`GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
YGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertAssert?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityiGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:output:0iGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:output:0iGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeiGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:output:0iGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shapeiGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityZ^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
]GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentitydGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0X^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
WGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOpZ^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
]graphregularization_graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1fGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
YGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertYGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8400

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????&& "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????KK :W S
/
_output_shapes
:?????????KK 
 
_user_specified_nameinputs
?
?
'__inference_conv2d_2_layer_call_fn_8369

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????KK `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????KK
 
_user_specified_nameinputs
?
?
fgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_7350?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
l
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholdern
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_1n
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_2
k
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1

agraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityb^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identityngraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1pgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?4
?
D__inference_sequential_layer_call_and_return_conditional_losses_8032
inputs_image?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpe
CastCastinputs_image*

DstT0*

SrcT0*1
_output_shapes
:????????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2DCast:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?
J
.__inference_max_pooling2d_1_layer_call_fn_8350

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8320

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_6105	
image%
conv2d_6075:
conv2d_6077:'
conv2d_1_6081:
conv2d_1_6083:'
conv2d_2_6087: 
conv2d_2_6089: 

dense_6094:
??@

dense_6096:@
dense_1_6099:@
dense_1_6101:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallimageconv2d_6075conv2d_6077*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5736?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_6081conv2d_1_6083*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_6087conv2d_2_6089*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&& * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5800?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_6094
dense_6096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5813?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6099dense_1_6101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5830w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameimage
?
?
&__inference_dense_1_layer_call_fn_8440

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5830o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
2__inference_GraphRegularization_layer_call_fn_7056
inputs_nl_nbr_0_image
inputs_nl_nbr_0_weight
inputs_nl_nbr_1_image
inputs_nl_nbr_1_weight
inputs_nl_nbr_2_image
inputs_nl_nbr_2_weight
inputs_image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_nl_nbr_0_imageinputs_nl_nbr_0_weightinputs_nl_nbr_1_imageinputs_nl_nbr_1_weightinputs_nl_nbr_2_imageinputs_nl_nbr_2_weightinputs_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:x t
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_0_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_0_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_1_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_1_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_2_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_2_weight:_[
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?!
?	
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_7297o
kgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapel
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?

ygraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIf?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch?R?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7306*
output_shapes
: *?
then_branch?R?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7305?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_8300

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>mean_squared_error_assert_broadcastable_AssertGuard_false_6326~
zmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identity
t
pmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_weights_shapes
omean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_values_shapep
lmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_scalar
B
>mean_squared_error_assert_broadcastable_assertguard_identity_1
??:mean_squared_error/assert_broadcastable/AssertGuard/Assert?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*
valueB B
inputs_2:0?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertzmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identityJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:output:0pmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_weights_shapeJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:output:0omean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_values_shapeJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:output:0lmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentityzmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identity;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
>mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityEmean_squared_error/assert_broadcastable/AssertGuard/Identity:output:09^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
>mean_squared_error_assert_broadcastable_assertguard_identity_1Gmean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2x
:mean_squared_error/assert_broadcastable/AssertGuard/Assert:mean_squared_error/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
Lgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_7203Q
Mgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder
?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_values_rank?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_weights_rank?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeN
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
hgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_values_rank?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
[graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIflgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_values_shape?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_weights_shapelgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *y
else_branchjRh
fgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7212*
output_shapes
: *x
then_branchiRg
egraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7211?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitydgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitySgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
fgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_7699?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
l
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholdern
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_1n
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_2
k
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1

agraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityb^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identityngraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1pgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
[mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_6419?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
a
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholderc
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_1c
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_2
`
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
t
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identitycmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1emean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?	
?
@mean_squared_error_assert_broadcastable_is_valid_shape_true_6271u
qmean_squared_error_assert_broadcastable_is_valid_shape_identity_mean_squared_error_assert_broadcastable_is_scalar
F
Bmean_squared_error_assert_broadcastable_is_valid_shape_placeholderH
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_1H
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_2H
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_3C
?mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentityqmean_squared_error_assert_broadcastable_is_valid_shape_identity_mean_squared_error_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
?mean_squared_error_assert_broadcastable_is_valid_shape_identityHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?,
?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_7351?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
k
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
??cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2graph_loss/mean_squared_error/num_present/Select:0?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identitysgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapesgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shapesgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityd^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identityngraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0b^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
agraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpd^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1pgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assertcgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_5991

inputs%
conv2d_5961:
conv2d_5963:'
conv2d_1_5967:
conv2d_1_5969:'
conv2d_2_5973: 
conv2d_2_5975: 

dense_5980:
??@

dense_5982:@
dense_1_5985:@
dense_1_5987:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5961conv2d_5963*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5736?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5967conv2d_1_5969*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_5973conv2d_2_5975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&& * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5800?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5980
dense_5982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5813?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_5985dense_1_5987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5830w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Lgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_7552Q
Mgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder
?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_values_rank?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_weights_rank?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeN
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
hgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_values_rank?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
[graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIflgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_values_shape?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_assert_broadcastable_weights_shapelgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *y
else_branchjRh
fgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7561*
output_shapes
: *x
then_branchiRg
egraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7560?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitydgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitySgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?;
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6477

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6)
sequential_6185:
sequential_6187:)
sequential_6189:
sequential_6191:)
sequential_6193: 
sequential_6195: #
sequential_6197:
??@
sequential_6199:@!
sequential_6201:@
sequential_6203:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?"graph_loss/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1=
ShapeShapeinputs_6*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPackinputsinputs_2inputs_4*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:x
stack_1Packinputs_1inputs_3inputs_5*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:??????????
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputs_6sequential_6185sequential_6187sequential_6189sequential_6191sequential_6193sequential_6195sequential_6197sequential_6199sequential_6201sequential_6203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6184?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallReshape:output:0sequential_6185sequential_6187sequential_6189sequential_6191sequential_6193sequential_6195sequential_6197sequential_6199sequential_6201sequential_6203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6184?
"graph_loss/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
mulMulmul/x:output:0+graph_loss/StatefulPartitionedCall:output:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: |

Identity_1Identity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1#^graph_loss/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12H
"graph_loss/StatefulPartitionedCall"graph_loss/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?	
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_7646o
kgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapel
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?

ygraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIf?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch?R?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7655*
output_shapes
: *?
then_branch?R?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7654?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?(
?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_6420?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
`
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
??Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityhmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapehmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shapehmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identitycmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0W^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1emean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
ymean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_6375~
zmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
|mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
{
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
)__inference_graph_loss_layer_call_fn_8039
inputs_0
inputs_1
inputs_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?4
?

{GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_5629?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar

{graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
??wGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFGraphRegularization/graph_loss/mean_squared_error/num_present/Select:0?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*V
valueMBK BEGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:0?
~GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
wGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
yGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityx^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
{GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0v^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
uGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpx^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
{graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
wGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertwGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_6072	
image%
conv2d_6042:
conv2d_6044:'
conv2d_1_6048:
conv2d_1_6050:'
conv2d_2_6054: 
conv2d_2_6056: 

dense_6061:
??@

dense_6063:@
dense_1_6066:@
dense_1_6068:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallimageconv2d_6042conv2d_6044*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5736?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_6048conv2d_1_6050*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_6054conv2d_2_6056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&& * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5800?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_6061
dense_6063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5813?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6066dense_1_6068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5830w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameimage
?
?
?GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_5584?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?4
?
D__inference_sequential_layer_call_and_return_conditional_losses_7987
inputs_image?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpe
CastCastinputs_image*

DstT0*

SrcT0*1
_output_shapes
:????????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2DCast:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?
?
Amean_squared_error_assert_broadcastable_is_valid_shape_false_8094F
Bmean_squared_error_assert_broadcastable_is_valid_shape_placeholder
?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_values_rank?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_weights_rank?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_values_shape?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_weights_shapeC
?mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_values_rank?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
Pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfamean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_values_shape?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_assert_broadcastable_weights_shapeamean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *n
else_branch_R]
[mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_8103*
output_shapes
: *m
then_branch^R\
Zmean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_8102?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
?mean_squared_error_assert_broadcastable_is_valid_shape_identityHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
2__inference_GraphRegularization_layer_call_fn_6791
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnl_nbr_0_imagenl_nbr_0_weightnl_nbr_1_imagenl_nbr_1_weightnl_nbr_2_imagenl_nbr_2_weightimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?&
?
~GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_5575?
graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder
?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
|graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIf?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch?R?
?GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_5584*
output_shapes
: *?
then_branch?R?
?GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_5583?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
|GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
|graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
2__inference_GraphRegularization_layer_call_fn_6505
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnl_nbr_0_imagenl_nbr_0_weightnl_nbr_1_imagenl_nbr_1_weightnl_nbr_2_imagenl_nbr_2_weightimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?
?
>mean_squared_error_assert_broadcastable_AssertGuard_false_8148~
zmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identity
t
pmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_weights_shapes
omean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_values_shapep
lmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_scalar
B
>mean_squared_error_assert_broadcastable_assertguard_identity_1
??:mean_squared_error/assert_broadcastable/AssertGuard/Assert?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*
valueB B
inputs/2:0?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertzmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identityJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:output:0pmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_weights_shapeJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:output:0Jmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:output:0omean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_values_shapeJmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:output:0lmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentityzmean_squared_error_assert_broadcastable_assertguard_assert_mean_squared_error_assert_broadcastable_is_valid_shape_identity;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
>mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityEmean_squared_error/assert_broadcastable/AssertGuard/Identity:output:09^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
>mean_squared_error_assert_broadcastable_assertguard_identity_1Gmean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2x
:mean_squared_error/assert_broadcastable/AssertGuard/Assert:mean_squared_error/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?)
?
Zmean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_6280?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_assert_broadcastable_values_shape?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_assert_broadcastable_weights_shape`
\mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
]
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_assert_broadcastable_values_shapeymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillzmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_assert_broadcastable_weights_shape{mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationwmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEquallmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0{mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityemean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
)__inference_sequential_layer_call_fn_7779

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5837o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?-
?
egraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7211?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_assert_broadcastable_weights_shapek
ggraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
h
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
{graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
wgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_assert_broadcastable_values_shape?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
vgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
xgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
sgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
}graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ygraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
}graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
ngraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
lgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitypgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
Kgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_7551?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
Q
Mgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholderS
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_1S
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_2S
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_3N
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitySgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
$__inference_dense_layer_call_fn_8420

inputs
unknown:
??@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5813o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_8451

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
zGraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_5490
{graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
}graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
|
xgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
xGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
xgraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?,
?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_7700?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
k
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
??cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2graph_loss/mean_squared_error/num_present/Select:0?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
jgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identitysgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapesgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:output:0sgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shapesgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityd^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identityngraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0b^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
agraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpd^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1pgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
cgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assertcgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
Kgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_7202?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
Q
Mgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholderS
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_1S
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_2S
Ograph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder_3N
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
Jgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitySgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
Hgraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_7256?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
N
Jgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholderP
Lgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_1P
Lgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_2
M
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
a
Cgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityD^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Igraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityPgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1Rgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?

?
)__inference_sequential_layer_call_fn_6039	
image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameimage
?=
?
?GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_5583?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
igraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_7645?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
o
kgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholderq
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_1q
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_2q
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_3l
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_8411

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&& :W S
/
_output_shapes
:?????????&& 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6984
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnl_nbr_0_imagenl_nbr_0_weightnl_nbr_1_imagenl_nbr_1_weightnl_nbr_2_imagenl_nbr_2_weightimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_5682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?3
?
D__inference_sequential_layer_call_and_return_conditional_losses_7942

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
Igraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_7606?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
M
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
??Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*
valueB BReshape_1:0?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?	
Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertAssert?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_values_shapeUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityF^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
Igraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityPgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0D^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Cgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOpF^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1Rgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertEgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?4
?
D__inference_sequential_layer_call_and_return_conditional_losses_6585

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp_
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:????????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2DCast:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_8310

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_2_layer_call_fn_8385

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5715?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_8431

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?(
?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_8242?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
`
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
??Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityhmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapehmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:output:0hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shapehmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:output:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_assert_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identitycmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0W^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1emean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
`GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_5481e
agraphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_placeholder
?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_rank?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_rank?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeb
^graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
|GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_rank?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?	
oGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIf?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_values_shape?graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_graphregularization_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch~R|
zGraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_5490*
output_shapes
: *?
then_branch}R{
yGraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_5489?
xGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityxGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
^graphregularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identitygGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_6366d
`mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder
?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapea
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?	
nmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch}R{
ymean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_6375*
output_shapes
: *?
then_branch|Rz
xmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_6374?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_5830

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
Igraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_7257?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_scalar
M
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
??Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*
valueB BReshape_1:0?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
Lgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?	
Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertAssert?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_weights_shapeUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:output:0Ugraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_values_shapeUgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:output:0?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_scalar*
T
2	
*
_output_shapes
 ?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_assertguard_assert_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityF^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*
_output_shapes
: ?
Igraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityPgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0D^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Cgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOpF^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1Rgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: 2?
Egraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/AssertEgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Assert: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_5736

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
ymean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_8197~
zmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder?
|mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
{
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?<
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6866
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image)
sequential_6816:
sequential_6818:)
sequential_6820:
sequential_6822:)
sequential_6824: 
sequential_6826: #
sequential_6828:
??@
sequential_6830:@!
sequential_6832:@
sequential_6834:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?"graph_loss/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1:
ShapeShapeimage*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPacknl_nbr_0_imagenl_nbr_1_imagenl_nbr_2_image*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
stack_1Packnl_nbr_0_weightnl_nbr_1_weightnl_nbr_2_weight*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:??????????
"sequential/StatefulPartitionedCallStatefulPartitionedCallimagesequential_6816sequential_6818sequential_6820sequential_6822sequential_6824sequential_6826sequential_6828sequential_6830sequential_6832sequential_6834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6184?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallReshape:output:0sequential_6816sequential_6818sequential_6820sequential_6822sequential_6824sequential_6826sequential_6828sequential_6830sequential_6832sequential_6834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6184?
"graph_loss/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
mulMulmul/x:output:0+graph_loss/StatefulPartitionedCall:output:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: |

Identity_1Identity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1#^graph_loss/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12H
"graph_loss/StatefulPartitionedCall"graph_loss/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?

?
)__inference_sequential_layer_call_fn_7854
inputs_image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?-
?
egraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7560?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_assert_broadcastable_weights_shapek
ggraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
h
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
{graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
wgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_assert_broadcastable_values_shape?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
vgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
xgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
sgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
}graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
ygraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0|graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
}graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
ngraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
lgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitypgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
B
&__inference_flatten_layer_call_fn_8405

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5800b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&& :W S
/
_output_shapes
:?????????&& 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????KK i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????KK w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????KK
 
_user_specified_nameinputs
?
?
}GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_5574?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
?
graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_1?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_2?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_3?
|graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
|GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
|graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
D__inference_graph_loss_layer_call_and_return_conditional_losses_8280
inputs_0
inputs_1
inputs_2
identity??"assert_greater_equal/Assert/Assert?3mean_squared_error/assert_broadcastable/AssertGuard?Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard=
ShapeShapeinputs_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
floordivFloorDivstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B : ?
!assert_greater_equal/GreaterEqualGreaterEqualfloordiv:z:0assert_greater_equal/y:output:0*
T0*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: ?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: ?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:u
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*"
valueB Bx (floordiv:0) = ?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = ?
)assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:{
)assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*"
valueB Bx (floordiv:0) = ?
)assert_greater_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = ?
"assert_greater_equal/Assert/AssertAssert!assert_greater_equal/All:output:02assert_greater_equal/Assert/Assert/data_0:output:02assert_greater_equal/Assert/Assert/data_1:output:0floordiv:z:02assert_greater_equal/Assert/Assert/data_3:output:0assert_greater_equal/y:output:0*
T	
2*
_output_shapes
 d
Shape_2Shapeinputs_0#^assert_greater_equal/Assert/Assert*
T0*
_output_shapes
:?
strided_slice_2/stackConst#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
strided_slice_2/stack_1Const#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2/stack_2Const#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
range/startConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : r
range/deltaConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :y
rangeRangerange/start:output:0strided_slice_2:output:0range/delta:output:0*#
_output_shapes
:?????????~
ExpandDims/dimConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
valueB :
?????????s

ExpandDims
ExpandDimsrange:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????w
Tile/multiples/0Const#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :m
Tile/multiplesPackTile/multiples/0:output:0floordiv:z:0*
N*
T0*
_output_shapes
:u
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*0
_output_shapes
:??????????????????S
mulMulstrided_slice_2:output:0floordiv:z:0*
T0*
_output_shapes
: L
Reshape/shapePackmul:z:0*
N*
T0*
_output_shapes
:g
ReshapeReshapeTile:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????t
GatherV2/axisConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2inputs_0Reshape:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:??????????
$mean_squared_error/SquaredDifferenceSquaredDifferenceinputs_1GatherV2:output:0*
T0*'
_output_shapes
:?????????m
5mean_squared_error/assert_broadcastable/weights/shapeShapeinputs_2*
T0*
_output_shapes
:v
4mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :?
4mean_squared_error/assert_broadcastable/values/shapeShape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:u
3mean_squared_error/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
1mean_squared_error/assert_broadcastable/is_scalarEqual<mean_squared_error/assert_broadcastable/is_scalar/x:output:0=mean_squared_error/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
6mean_squared_error/assert_broadcastable/is_valid_shapeStatelessIf5mean_squared_error/assert_broadcastable/is_scalar:z:05mean_squared_error/assert_broadcastable/is_scalar:z:0<mean_squared_error/assert_broadcastable/values/rank:output:0=mean_squared_error/assert_broadcastable/weights/rank:output:0=mean_squared_error/assert_broadcastable/values/shape:output:0>mean_squared_error/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *T
else_branchERC
Amean_squared_error_assert_broadcastable_is_valid_shape_false_8094*
output_shapes
: *S
then_branchDRB
@mean_squared_error_assert_broadcastable_is_valid_shape_true_8093?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
-mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.~
/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=z
/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*
valueB B
inputs/2:0}
/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0z
/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
3mean_squared_error/assert_broadcastable/AssertGuardIfHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Hmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0>mean_squared_error/assert_broadcastable/weights/shape:output:0=mean_squared_error/assert_broadcastable/values/shape:output:05mean_squared_error/assert_broadcastable/is_scalar:z:0#^assert_greater_equal/Assert/Assert*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Q
else_branchBR@
>mean_squared_error_assert_broadcastable_AssertGuard_false_8148*
output_shapes
: *P
then_branchAR?
=mean_squared_error_assert_broadcastable_AssertGuard_true_8147?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity<mean_squared_error/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
mean_squared_error/MulMul(mean_squared_error/SquaredDifference:z:0inputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
mean_squared_error/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       }
mean_squared_error/SumSummean_squared_error/Mul:z:0!mean_squared_error/Const:output:0*
T0*
_output_shapes
: ?
&mean_squared_error/num_present/Equal/yConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    ?
$mean_squared_error/num_present/EqualEqualinputs_2/mean_squared_error/num_present/Equal/y:output:0*
T0*'
_output_shapes
:??????????
)mean_squared_error/num_present/zeros_like	ZerosLikeinputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
.mean_squared_error/num_present/ones_like/ShapeShapeinputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
.mean_squared_error/num_present/ones_like/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
(mean_squared_error/num_present/ones_likeFill7mean_squared_error/num_present/ones_like/Shape:output:07mean_squared_error/num_present/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
%mean_squared_error/num_present/SelectSelect(mean_squared_error/num_present/Equal:z:0-mean_squared_error/num_present/zeros_like:y:01mean_squared_error/num_present/ones_like:output:0*
T0*'
_output_shapes
:??????????
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeShape.mean_squared_error/num_present/Select:output:0*
T0*
_output_shapes
:?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape(mean_squared_error/SquaredDifference:z:0=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/x:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shapeStatelessIfSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *r
else_branchcRa
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_8188*
output_shapes
: *q
then_branchbR`
^mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_8187?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardIffmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:04^mean_squared_error/assert_broadcastable/AssertGuard*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_8242*
output_shapes
: *n
then_branch_R]
[mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_8241?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape(mean_squared_error/SquaredDifference:z:0=^mean_squared_error/assert_broadcastable/AssertGuard/Identity[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
:mean_squared_error/num_present/broadcast_weights/ones_likeFillImean_squared_error/num_present/broadcast_weights/ones_like/Shape:output:0Imean_squared_error/num_present/broadcast_weights/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
0mean_squared_error/num_present/broadcast_weightsMul.mean_squared_error/num_present/Select:output:0Cmean_squared_error/num_present/broadcast_weights/ones_like:output:0*
T0*'
_output_shapes
:??????????
$mean_squared_error/num_present/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
mean_squared_error/num_presentSum4mean_squared_error/num_present/broadcast_weights:z:0-mean_squared_error/num_present/Const:output:0*
T0*
_output_shapes
: ?
mean_squared_error/RankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/range/startConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/range/deltaConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
mean_squared_error/rangeRange'mean_squared_error/range/start:output:0 mean_squared_error/Rank:output:0'mean_squared_error/range/delta:output:0*
_output_shapes
: ?
mean_squared_error/Sum_1Summean_squared_error/Sum:output:0!mean_squared_error/range:output:0*
T0*
_output_shapes
: ?
mean_squared_error/valueDivNoNan!mean_squared_error/Sum_1:output:0'mean_squared_error/num_present:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@]
mul_1Mulmean_squared_error/value:z:0mul_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp#^assert_greater_equal/Assert/Assert4^mean_squared_error/assert_broadcastable/AssertGuardR^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????2H
"assert_greater_equal/Assert/Assert"assert_greater_equal/Assert/Assert2j
3mean_squared_error/assert_broadcastable/AssertGuard3mean_squared_error/assert_broadcastable/AssertGuard2?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
Hgraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_7605?
?graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identity
N
Jgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholderP
Lgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_1P
Lgraph_loss_mean_squared_error_assert_broadcastable_assertguard_placeholder_2
M
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1
a
Cgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_identityD^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
Igraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityPgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
Igraph_loss_mean_squared_error_assert_broadcastable_assertguard_identity_1Rgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?	
?
[mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_6281`
\mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderb
^mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
]
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5691

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
[mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_8103`
\mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderb
^mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
]
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
igraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_7296?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
o
kgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholderq
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_1q
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_2q
mgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_3l
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
hgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
^mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_8187?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar
d
`mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholderf
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_1f
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_2f
bmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder_3a
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?

?
=mean_squared_error_assert_broadcastable_AssertGuard_true_8147?
|mean_squared_error_assert_broadcastable_assertguard_identity_mean_squared_error_assert_broadcastable_is_valid_shape_identity
C
?mean_squared_error_assert_broadcastable_assertguard_placeholderE
Amean_squared_error_assert_broadcastable_assertguard_placeholder_1E
Amean_squared_error_assert_broadcastable_assertguard_placeholder_2
B
>mean_squared_error_assert_broadcastable_assertguard_identity_1
V
8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity|mean_squared_error_assert_broadcastable_assertguard_identity_mean_squared_error_assert_broadcastable_is_valid_shape_identity9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
>mean_squared_error/assert_broadcastable/AssertGuard/Identity_1IdentityEmean_squared_error/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
>mean_squared_error_assert_broadcastable_assertguard_identity_1Gmean_squared_error/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
H
,__inference_max_pooling2d_layer_call_fn_8305

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5691?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?4
?
D__inference_sequential_layer_call_and_return_conditional_losses_6184

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp_
CastCastinputs*

DstT0*

SrcT0*1
_output_shapes
:????????????
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2DCast:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
)__inference_sequential_layer_call_fn_7829
inputs_image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_5837

inputs%
conv2d_5737:
conv2d_5739:'
conv2d_1_5760:
conv2d_1_5762:'
conv2d_2_5783: 
conv2d_2_5785: 

dense_5814:
??@

dense_5816:@
dense_1_5831:@
dense_1_5833:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5737conv2d_5739*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5736?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5746?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5760conv2d_1_5762*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5759?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5769?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_5783conv2d_2_5785*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5782?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&& * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_5792?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5800?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5814
dense_5816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5813?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_5831dense_1_5833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5830w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458

inputs
inputs_1
inputs_2
identity??"assert_greater_equal/Assert/Assert?3mean_squared_error/assert_broadcastable/AssertGuard?Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard=
ShapeShapeinputs_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
floordivFloorDivstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: X
assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B : ?
!assert_greater_equal/GreaterEqualGreaterEqualfloordiv:z:0assert_greater_equal/y:output:0*
T0*
_output_shapes
: [
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: ?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: ?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:u
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*"
valueB Bx (floordiv:0) = ?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = ?
)assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:{
)assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*"
valueB Bx (floordiv:0) = ?
)assert_greater_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = ?
"assert_greater_equal/Assert/AssertAssert!assert_greater_equal/All:output:02assert_greater_equal/Assert/Assert/data_0:output:02assert_greater_equal/Assert/Assert/data_1:output:0floordiv:z:02assert_greater_equal/Assert/Assert/data_3:output:0assert_greater_equal/y:output:0*
T	
2*
_output_shapes
 b
Shape_2Shapeinputs#^assert_greater_equal/Assert/Assert*
T0*
_output_shapes
:?
strided_slice_2/stackConst#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
strided_slice_2/stack_1Const#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2/stack_2Const#^assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
range/startConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : r
range/deltaConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :y
rangeRangerange/start:output:0strided_slice_2:output:0range/delta:output:0*#
_output_shapes
:?????????~
ExpandDims/dimConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
valueB :
?????????s

ExpandDims
ExpandDimsrange:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????w
Tile/multiples/0Const#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :m
Tile/multiplesPackTile/multiples/0:output:0floordiv:z:0*
N*
T0*
_output_shapes
:u
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*0
_output_shapes
:??????????????????S
mulMulstrided_slice_2:output:0floordiv:z:0*
T0*
_output_shapes
: L
Reshape/shapePackmul:z:0*
N*
T0*
_output_shapes
:g
ReshapeReshapeTile:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????t
GatherV2/axisConst#^assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2inputsReshape:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:??????????
$mean_squared_error/SquaredDifferenceSquaredDifferenceinputs_1GatherV2:output:0*
T0*'
_output_shapes
:?????????m
5mean_squared_error/assert_broadcastable/weights/shapeShapeinputs_2*
T0*
_output_shapes
:v
4mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :?
4mean_squared_error/assert_broadcastable/values/shapeShape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:u
3mean_squared_error/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
1mean_squared_error/assert_broadcastable/is_scalarEqual<mean_squared_error/assert_broadcastable/is_scalar/x:output:0=mean_squared_error/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
6mean_squared_error/assert_broadcastable/is_valid_shapeStatelessIf5mean_squared_error/assert_broadcastable/is_scalar:z:05mean_squared_error/assert_broadcastable/is_scalar:z:0<mean_squared_error/assert_broadcastable/values/rank:output:0=mean_squared_error/assert_broadcastable/weights/rank:output:0=mean_squared_error/assert_broadcastable/values/shape:output:0>mean_squared_error/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *T
else_branchERC
Amean_squared_error_assert_broadcastable_is_valid_shape_false_6272*
output_shapes
: *S
then_branchDRB
@mean_squared_error_assert_broadcastable_is_valid_shape_true_6271?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
-mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.~
/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=z
/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*
valueB B
inputs_2:0}
/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0z
/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
3mean_squared_error/assert_broadcastable/AssertGuardIfHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Hmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0>mean_squared_error/assert_broadcastable/weights/shape:output:0=mean_squared_error/assert_broadcastable/values/shape:output:05mean_squared_error/assert_broadcastable/is_scalar:z:0#^assert_greater_equal/Assert/Assert*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Q
else_branchBR@
>mean_squared_error_assert_broadcastable_AssertGuard_false_6326*
output_shapes
: *P
then_branchAR?
=mean_squared_error_assert_broadcastable_AssertGuard_true_6325?
<mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity<mean_squared_error/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
mean_squared_error/MulMul(mean_squared_error/SquaredDifference:z:0inputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
mean_squared_error/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       }
mean_squared_error/SumSummean_squared_error/Mul:z:0!mean_squared_error/Const:output:0*
T0*
_output_shapes
: ?
&mean_squared_error/num_present/Equal/yConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    ?
$mean_squared_error/num_present/EqualEqualinputs_2/mean_squared_error/num_present/Equal/y:output:0*
T0*'
_output_shapes
:??????????
)mean_squared_error/num_present/zeros_like	ZerosLikeinputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
.mean_squared_error/num_present/ones_like/ShapeShapeinputs_2=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
.mean_squared_error/num_present/ones_like/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
(mean_squared_error/num_present/ones_likeFill7mean_squared_error/num_present/ones_like/Shape:output:07mean_squared_error/num_present/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
%mean_squared_error/num_present/SelectSelect(mean_squared_error/num_present/Equal:z:0-mean_squared_error/num_present/zeros_like:y:01mean_squared_error/num_present/ones_like:output:0*
T0*'
_output_shapes
:??????????
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeShape.mean_squared_error/num_present/Select:output:0*
T0*
_output_shapes
:?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape(mean_squared_error/SquaredDifference:z:0=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/x:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shapeStatelessIfSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *r
else_branchcRa
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_6366*
output_shapes
: *q
then_branchbR`
^mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_6365?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'mean_squared_error/num_present/Select:0?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*7
value.B, B&mean_squared_error/SquaredDifference:0?
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardIffmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:04^mean_squared_error/assert_broadcastable/AssertGuard*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_6420*
output_shapes
: *n
then_branch_R]
[mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_6419?
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape(mean_squared_error/SquaredDifference:z:0=^mean_squared_error/assert_broadcastable/AssertGuard/Identity[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
:mean_squared_error/num_present/broadcast_weights/ones_likeFillImean_squared_error/num_present/broadcast_weights/ones_like/Shape:output:0Imean_squared_error/num_present/broadcast_weights/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
0mean_squared_error/num_present/broadcast_weightsMul.mean_squared_error/num_present/Select:output:0Cmean_squared_error/num_present/broadcast_weights/ones_like:output:0*
T0*'
_output_shapes
:??????????
$mean_squared_error/num_present/ConstConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
mean_squared_error/num_presentSum4mean_squared_error/num_present/broadcast_weights:z:0-mean_squared_error/num_present/Const:output:0*
T0*
_output_shapes
: ?
mean_squared_error/RankConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/range/startConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/range/deltaConst=^mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
mean_squared_error/rangeRange'mean_squared_error/range/start:output:0 mean_squared_error/Rank:output:0'mean_squared_error/range/delta:output:0*
_output_shapes
: ?
mean_squared_error/Sum_1Summean_squared_error/Sum:output:0!mean_squared_error/range:output:0*
T0*
_output_shapes
: ?
mean_squared_error/valueDivNoNan!mean_squared_error/Sum_1:output:0'mean_squared_error/num_present:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@]
mul_1Mulmean_squared_error/value:z:0mul_1/y:output:0*
T0*
_output_shapes
: G
IdentityIdentity	mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp#^assert_greater_equal/Assert/Assert4^mean_squared_error/assert_broadcastable/AssertGuardR^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????2H
"assert_greater_equal/Assert/Assert"assert_greater_equal/Assert/Assert2j
3mean_squared_error/assert_broadcastable/AssertGuard3mean_squared_error/assert_broadcastable/AssertGuard2?
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_GraphRegularization_layer_call_fn_7020
inputs_nl_nbr_0_image
inputs_nl_nbr_0_weight
inputs_nl_nbr_1_image
inputs_nl_nbr_1_weight
inputs_nl_nbr_2_image
inputs_nl_nbr_2_weight
inputs_image!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5:
??@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9: 

unknown_10: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_nl_nbr_0_imageinputs_nl_nbr_0_weightinputs_nl_nbr_1_imageinputs_nl_nbr_1_weightinputs_nl_nbr_2_imageinputs_nl_nbr_2_weightinputs_imageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:x t
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_0_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_0_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_1_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_1_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_2_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_2_weight:_[
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?)
?
Zmean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_8102?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_assert_broadcastable_values_shape?
?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_assert_broadcastable_weights_shape`
\mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
]
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_assert_broadcastable_values_shapeymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillzmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_assert_broadcastable_weights_shape{mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationwmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEquallmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0{mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentityemean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
Ymean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitybmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?3
?
D__inference_sequential_layer_call_and_return_conditional_losses_7898

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: 8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????l
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?6
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7654?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
zGraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_5628?
?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
|graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder?
~graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_1?
~graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_placeholder_2

{graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1
?
uGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp*
_output_shapes
 ?
yGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentity?graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityv^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*
_output_shapes
: ?
{GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1Identity?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "?
{graphregularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_assertguard_identity_1?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: ::: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?<
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6941
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image)
sequential_6891:
sequential_6893:)
sequential_6895:
sequential_6897:)
sequential_6899: 
sequential_6901: #
sequential_6903:
??@
sequential_6905:@!
sequential_6907:@
sequential_6909:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?"graph_loss/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1:
ShapeShapeimage*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPacknl_nbr_0_imagenl_nbr_1_imagenl_nbr_2_image*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
stack_1Packnl_nbr_0_weightnl_nbr_1_weightnl_nbr_2_weight*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:??????????
"sequential/StatefulPartitionedCallStatefulPartitionedCallimagesequential_6891sequential_6893sequential_6895sequential_6897sequential_6899sequential_6901sequential_6903sequential_6905sequential_6907sequential_6909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6585?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallReshape:output:0sequential_6891sequential_6893sequential_6895sequential_6897sequential_6899sequential_6901sequential_6903sequential_6905sequential_6907sequential_6909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6585?
"graph_loss/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_graph_loss_layer_call_and_return_conditional_losses_6458J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
mulMulmul/x:output:0+graph_loss/StatefulPartitionedCall:output:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: |

Identity_1Identity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1#^graph_loss/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12H
"graph_loss/StatefulPartitionedCall"graph_loss/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?	
?
@mean_squared_error_assert_broadcastable_is_valid_shape_true_8093u
qmean_squared_error_assert_broadcastable_is_valid_shape_identity_mean_squared_error_assert_broadcastable_is_scalar
F
Bmean_squared_error_assert_broadcastable_is_valid_shape_placeholderH
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_1H
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_2H
Dmean_squared_error_assert_broadcastable_is_valid_shape_placeholder_3C
?mean_squared_error_assert_broadcastable_is_valid_shape_identity
?
?mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentityqmean_squared_error_assert_broadcastable_is_valid_shape_identity_mean_squared_error_assert_broadcastable_is_scalar*
T0
*
_output_shapes
: "?
?mean_squared_error_assert_broadcastable_is_valid_shape_identityHmean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?3
?
xmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_6374?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape~
zmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
{
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
??
?
__inference__wrapped_model_5682
nl_nbr_0_image
nl_nbr_0_weight
nl_nbr_1_image
nl_nbr_1_weight
nl_nbr_2_image
nl_nbr_2_weight	
image^
Dgraphregularization_sequential_conv2d_conv2d_readvariableop_resource:S
Egraphregularization_sequential_conv2d_biasadd_readvariableop_resource:`
Fgraphregularization_sequential_conv2d_1_conv2d_readvariableop_resource:U
Ggraphregularization_sequential_conv2d_1_biasadd_readvariableop_resource:`
Fgraphregularization_sequential_conv2d_2_conv2d_readvariableop_resource: U
Ggraphregularization_sequential_conv2d_2_biasadd_readvariableop_resource: W
Cgraphregularization_sequential_dense_matmul_readvariableop_resource:
??@R
Dgraphregularization_sequential_dense_biasadd_readvariableop_resource:@W
Egraphregularization_sequential_dense_1_matmul_readvariableop_resource:@T
Fgraphregularization_sequential_dense_1_biasadd_readvariableop_resource::
0graphregularization_assignaddvariableop_resource: <
2graphregularization_assignaddvariableop_1_resource: 
identity??'GraphRegularization/AssignAddVariableOp?)GraphRegularization/AssignAddVariableOp_1?-GraphRegularization/div_no_nan/ReadVariableOp?/GraphRegularization/div_no_nan/ReadVariableOp_1?AGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert?RGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard?pGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard?<GraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOp?>GraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOp?;GraphRegularization/sequential/conv2d/Conv2D/ReadVariableOp?=GraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOp?>GraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOp?@GraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOp?=GraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOp??GraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOp?>GraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOp?@GraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOp?=GraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOp??GraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOp?;GraphRegularization/sequential/dense/BiasAdd/ReadVariableOp?=GraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOp?:GraphRegularization/sequential/dense/MatMul/ReadVariableOp?<GraphRegularization/sequential/dense/MatMul_1/ReadVariableOp?=GraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOp??GraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOp?<GraphRegularization/sequential/dense_1/MatMul/ReadVariableOp?>GraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOpN
GraphRegularization/ShapeShapeimage*
T0*
_output_shapes
:l
GraphRegularization/ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
'GraphRegularization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)GraphRegularization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)GraphRegularization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!GraphRegularization/strided_sliceStridedSlice"GraphRegularization/Shape:output:00GraphRegularization/strided_slice/stack:output:02GraphRegularization/strided_slice/stack_1:output:02GraphRegularization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maska
GraphRegularization/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GraphRegularization/concatConcatV2"GraphRegularization/Const:output:0*GraphRegularization/strided_slice:output:0(GraphRegularization/concat/axis:output:0*
N*
T0*
_output_shapes
:?
GraphRegularization/stackPacknl_nbr_0_imagenl_nbr_1_imagenl_nbr_2_image*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axis?
GraphRegularization/ReshapeReshape"GraphRegularization/stack:output:0#GraphRegularization/concat:output:0*
T0*1
_output_shapes
:???????????n
GraphRegularization/Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%GraphRegularization/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:c
!GraphRegularization/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GraphRegularization/concat_1ConcatV2$GraphRegularization/Const_1:output:0.GraphRegularization/concat_1/values_1:output:0*GraphRegularization/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
GraphRegularization/stack_1Packnl_nbr_0_weightnl_nbr_1_weightnl_nbr_2_weight*
N*
T0*+
_output_shapes
:?????????*

axis?
GraphRegularization/Reshape_1Reshape$GraphRegularization/stack_1:output:0%GraphRegularization/concat_1:output:0*
T0*'
_output_shapes
:?????????}
#GraphRegularization/sequential/CastCastimage*

DstT0*

SrcT0*1
_output_shapes
:????????????
;GraphRegularization/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpDgraphregularization_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
,GraphRegularization/sequential/conv2d/Conv2DConv2D'GraphRegularization/sequential/Cast:y:0CGraphRegularization/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
<GraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpEgraphregularization_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-GraphRegularization/sequential/conv2d/BiasAddBiasAdd5GraphRegularization/sequential/conv2d/Conv2D:output:0DGraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
*GraphRegularization/sequential/conv2d/ReluRelu6GraphRegularization/sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
4GraphRegularization/sequential/max_pooling2d/MaxPoolMaxPool8GraphRegularization/sequential/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
=GraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpFgraphregularization_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.GraphRegularization/sequential/conv2d_1/Conv2DConv2D=GraphRegularization/sequential/max_pooling2d/MaxPool:output:0EGraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
>GraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGgraphregularization_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
/GraphRegularization/sequential/conv2d_1/BiasAddBiasAdd7GraphRegularization/sequential/conv2d_1/Conv2D:output:0FGraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
,GraphRegularization/sequential/conv2d_1/ReluRelu8GraphRegularization/sequential/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
6GraphRegularization/sequential/max_pooling2d_1/MaxPoolMaxPool:GraphRegularization/sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
=GraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpFgraphregularization_sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
.GraphRegularization/sequential/conv2d_2/Conv2DConv2D?GraphRegularization/sequential/max_pooling2d_1/MaxPool:output:0EGraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
>GraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGgraphregularization_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
/GraphRegularization/sequential/conv2d_2/BiasAddBiasAdd7GraphRegularization/sequential/conv2d_2/Conv2D:output:0FGraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
,GraphRegularization/sequential/conv2d_2/ReluRelu8GraphRegularization/sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
6GraphRegularization/sequential/max_pooling2d_2/MaxPoolMaxPool:GraphRegularization/sequential/conv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
}
,GraphRegularization/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
.GraphRegularization/sequential/flatten/ReshapeReshape?GraphRegularization/sequential/max_pooling2d_2/MaxPool:output:05GraphRegularization/sequential/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
:GraphRegularization/sequential/dense/MatMul/ReadVariableOpReadVariableOpCgraphregularization_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
+GraphRegularization/sequential/dense/MatMulMatMul7GraphRegularization/sequential/flatten/Reshape:output:0BGraphRegularization/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
;GraphRegularization/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpDgraphregularization_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
,GraphRegularization/sequential/dense/BiasAddBiasAdd5GraphRegularization/sequential/dense/MatMul:product:0CGraphRegularization/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)GraphRegularization/sequential/dense/ReluRelu5GraphRegularization/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
<GraphRegularization/sequential/dense_1/MatMul/ReadVariableOpReadVariableOpEgraphregularization_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
-GraphRegularization/sequential/dense_1/MatMulMatMul7GraphRegularization/sequential/dense/Relu:activations:0DGraphRegularization/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
=GraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpFgraphregularization_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
.GraphRegularization/sequential/dense_1/BiasAddBiasAdd7GraphRegularization/sequential/dense_1/MatMul:product:0EGraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.GraphRegularization/sequential/dense_1/SoftmaxSoftmax7GraphRegularization/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%GraphRegularization/sequential/Cast_1Cast$GraphRegularization/Reshape:output:0*

DstT0*

SrcT0*1
_output_shapes
:????????????
=GraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOpReadVariableOpDgraphregularization_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.GraphRegularization/sequential/conv2d/Conv2D_1Conv2D)GraphRegularization/sequential/Cast_1:y:0EGraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
>GraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOpReadVariableOpEgraphregularization_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
/GraphRegularization/sequential/conv2d/BiasAdd_1BiasAdd7GraphRegularization/sequential/conv2d/Conv2D_1:output:0FGraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
,GraphRegularization/sequential/conv2d/Relu_1Relu8GraphRegularization/sequential/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
6GraphRegularization/sequential/max_pooling2d/MaxPool_1MaxPool:GraphRegularization/sequential/conv2d/Relu_1:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
?GraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOpFgraphregularization_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
0GraphRegularization/sequential/conv2d_1/Conv2D_1Conv2D?GraphRegularization/sequential/max_pooling2d/MaxPool_1:output:0GGraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
@GraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOpGgraphregularization_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
1GraphRegularization/sequential/conv2d_1/BiasAdd_1BiasAdd9GraphRegularization/sequential/conv2d_1/Conv2D_1:output:0HGraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
.GraphRegularization/sequential/conv2d_1/Relu_1Relu:GraphRegularization/sequential/conv2d_1/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
8GraphRegularization/sequential/max_pooling2d_1/MaxPool_1MaxPool<GraphRegularization/sequential/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
?GraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOpFgraphregularization_sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
0GraphRegularization/sequential/conv2d_2/Conv2D_1Conv2DAGraphRegularization/sequential/max_pooling2d_1/MaxPool_1:output:0GGraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
@GraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOpGgraphregularization_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
1GraphRegularization/sequential/conv2d_2/BiasAdd_1BiasAdd9GraphRegularization/sequential/conv2d_2/Conv2D_1:output:0HGraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
.GraphRegularization/sequential/conv2d_2/Relu_1Relu:GraphRegularization/sequential/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????KK ?
8GraphRegularization/sequential/max_pooling2d_2/MaxPool_1MaxPool<GraphRegularization/sequential/conv2d_2/Relu_1:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides

.GraphRegularization/sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"??????  ?
0GraphRegularization/sequential/flatten/Reshape_1ReshapeAGraphRegularization/sequential/max_pooling2d_2/MaxPool_1:output:07GraphRegularization/sequential/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
<GraphRegularization/sequential/dense/MatMul_1/ReadVariableOpReadVariableOpCgraphregularization_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
-GraphRegularization/sequential/dense/MatMul_1MatMul9GraphRegularization/sequential/flatten/Reshape_1:output:0DGraphRegularization/sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
=GraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOpDgraphregularization_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
.GraphRegularization/sequential/dense/BiasAdd_1BiasAdd7GraphRegularization/sequential/dense/MatMul_1:product:0EGraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+GraphRegularization/sequential/dense/Relu_1Relu7GraphRegularization/sequential/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????@?
>GraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOpEgraphregularization_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
/GraphRegularization/sequential/dense_1/MatMul_1MatMul9GraphRegularization/sequential/dense/Relu_1:activations:0FGraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
?GraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOpFgraphregularization_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
0GraphRegularization/sequential/dense_1/BiasAdd_1BiasAdd9GraphRegularization/sequential/dense_1/MatMul_1:product:0GGraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0GraphRegularization/sequential/dense_1/Softmax_1Softmax9GraphRegularization/sequential/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:??????????
$GraphRegularization/graph_loss/ShapeShape:GraphRegularization/sequential/dense_1/Softmax_1:softmax:0*
T0*
_output_shapes
:|
2GraphRegularization/graph_loss/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4GraphRegularization/graph_loss/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4GraphRegularization/graph_loss/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,GraphRegularization/graph_loss/strided_sliceStridedSlice-GraphRegularization/graph_loss/Shape:output:0;GraphRegularization/graph_loss/strided_slice/stack:output:0=GraphRegularization/graph_loss/strided_slice/stack_1:output:0=GraphRegularization/graph_loss/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&GraphRegularization/graph_loss/Shape_1Shape8GraphRegularization/sequential/dense_1/Softmax:softmax:0*
T0*
_output_shapes
:~
4GraphRegularization/graph_loss/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6GraphRegularization/graph_loss/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6GraphRegularization/graph_loss/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.GraphRegularization/graph_loss/strided_slice_1StridedSlice/GraphRegularization/graph_loss/Shape_1:output:0=GraphRegularization/graph_loss/strided_slice_1/stack:output:0?GraphRegularization/graph_loss/strided_slice_1/stack_1:output:0?GraphRegularization/graph_loss/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'GraphRegularization/graph_loss/floordivFloorDiv5GraphRegularization/graph_loss/strided_slice:output:07GraphRegularization/graph_loss/strided_slice_1:output:0*
T0*
_output_shapes
: w
5GraphRegularization/graph_loss/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B : ?
@GraphRegularization/graph_loss/assert_greater_equal/GreaterEqualGreaterEqual+GraphRegularization/graph_loss/floordiv:z:0>GraphRegularization/graph_loss/assert_greater_equal/y:output:0*
T0*
_output_shapes
: z
8GraphRegularization/graph_loss/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : ?
?GraphRegularization/graph_loss/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : ?
?GraphRegularization/graph_loss/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
9GraphRegularization/graph_loss/assert_greater_equal/rangeRangeHGraphRegularization/graph_loss/assert_greater_equal/range/start:output:0AGraphRegularization/graph_loss/assert_greater_equal/Rank:output:0HGraphRegularization/graph_loss/assert_greater_equal/range/delta:output:0*
_output_shapes
: ?
7GraphRegularization/graph_loss/assert_greater_equal/AllAllDGraphRegularization/graph_loss/assert_greater_equal/GreaterEqual:z:0BGraphRegularization/graph_loss/assert_greater_equal/range:output:0*
_output_shapes
: ?
@GraphRegularization/graph_loss/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
BGraphRegularization/graph_loss/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0x (GraphRegularization/graph_loss/floordiv:0) = ?
BGraphRegularization/graph_loss/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (GraphRegularization/graph_loss/assert_greater_equal/y:0) = ?
HGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
HGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0x (GraphRegularization/graph_loss/floordiv:0) = ?
HGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (GraphRegularization/graph_loss/assert_greater_equal/y:0) = ?
AGraphRegularization/graph_loss/assert_greater_equal/Assert/AssertAssert@GraphRegularization/graph_loss/assert_greater_equal/All:output:0QGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_0:output:0QGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_1:output:0+GraphRegularization/graph_loss/floordiv:z:0QGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert/data_3:output:0>GraphRegularization/graph_loss/assert_greater_equal/y:output:0*
T	
2*
_output_shapes
 ?
&GraphRegularization/graph_loss/Shape_2Shape8GraphRegularization/sequential/dense_1/Softmax:softmax:0B^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
T0*
_output_shapes
:?
4GraphRegularization/graph_loss/strided_slice_2/stackConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
6GraphRegularization/graph_loss/strided_slice_2/stack_1ConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
6GraphRegularization/graph_loss/strided_slice_2/stack_2ConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
.GraphRegularization/graph_loss/strided_slice_2StridedSlice/GraphRegularization/graph_loss/Shape_2:output:0=GraphRegularization/graph_loss/strided_slice_2/stack:output:0?GraphRegularization/graph_loss/strided_slice_2/stack_1:output:0?GraphRegularization/graph_loss/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*GraphRegularization/graph_loss/range/startConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
*GraphRegularization/graph_loss/range/deltaConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
$GraphRegularization/graph_loss/rangeRange3GraphRegularization/graph_loss/range/start:output:07GraphRegularization/graph_loss/strided_slice_2:output:03GraphRegularization/graph_loss/range/delta:output:0*#
_output_shapes
:??????????
-GraphRegularization/graph_loss/ExpandDims/dimConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
valueB :
??????????
)GraphRegularization/graph_loss/ExpandDims
ExpandDims-GraphRegularization/graph_loss/range:output:06GraphRegularization/graph_loss/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
/GraphRegularization/graph_loss/Tile/multiples/0ConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
-GraphRegularization/graph_loss/Tile/multiplesPack8GraphRegularization/graph_loss/Tile/multiples/0:output:0+GraphRegularization/graph_loss/floordiv:z:0*
N*
T0*
_output_shapes
:?
#GraphRegularization/graph_loss/TileTile2GraphRegularization/graph_loss/ExpandDims:output:06GraphRegularization/graph_loss/Tile/multiples:output:0*
T0*0
_output_shapes
:???????????????????
"GraphRegularization/graph_loss/mulMul7GraphRegularization/graph_loss/strided_slice_2:output:0+GraphRegularization/graph_loss/floordiv:z:0*
T0*
_output_shapes
: ?
,GraphRegularization/graph_loss/Reshape/shapePack&GraphRegularization/graph_loss/mul:z:0*
N*
T0*
_output_shapes
:?
&GraphRegularization/graph_loss/ReshapeReshape,GraphRegularization/graph_loss/Tile:output:05GraphRegularization/graph_loss/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
,GraphRegularization/graph_loss/GatherV2/axisConstB^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
'GraphRegularization/graph_loss/GatherV2GatherV28GraphRegularization/sequential/dense_1/Softmax:softmax:0/GraphRegularization/graph_loss/Reshape:output:05GraphRegularization/graph_loss/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:??????????
CGraphRegularization/graph_loss/mean_squared_error/SquaredDifferenceSquaredDifference:GraphRegularization/sequential/dense_1/Softmax_1:softmax:00GraphRegularization/graph_loss/GatherV2:output:0*
T0*'
_output_shapes
:??????????
TGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/shapeShape&GraphRegularization/Reshape_1:output:0*
T0*
_output_shapes
:?
SGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :?
SGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/values/shapeShapeGGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:?
RGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :?
RGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
PGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalarEqual[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalar/x:output:0\GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
UGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shapeStatelessIfTGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0TGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/values/rank:output:0\GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0\GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0]GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *s
else_branchdRb
`GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_5481*
output_shapes
: *r
then_branchcRa
_GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_5480?
^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentity^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
LGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
NGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
NGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*0
value'B% BGraphRegularization/Reshape_1:0?
NGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
NGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*V
valueMBK BEGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:0?
NGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
RGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuardIfgGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0gGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0]GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0\GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0TGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0B^GraphRegularization/graph_loss/assert_greater_equal/Assert/Assert*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *p
else_branchaR_
]GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_5535*
output_shapes
: *o
then_branch`R^
\GraphRegularization_graph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_5534?
[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentity[GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
5GraphRegularization/graph_loss/mean_squared_error/MulMulGGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:z:0&GraphRegularization/Reshape_1:output:0\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
7GraphRegularization/graph_loss/mean_squared_error/ConstConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
5GraphRegularization/graph_loss/mean_squared_error/SumSum9GraphRegularization/graph_loss/mean_squared_error/Mul:z:0@GraphRegularization/graph_loss/mean_squared_error/Const:output:0*
T0*
_output_shapes
: ?
EGraphRegularization/graph_loss/mean_squared_error/num_present/Equal/yConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    ?
CGraphRegularization/graph_loss/mean_squared_error/num_present/EqualEqual&GraphRegularization/Reshape_1:output:0NGraphRegularization/graph_loss/mean_squared_error/num_present/Equal/y:output:0*
T0*'
_output_shapes
:??????????
HGraphRegularization/graph_loss/mean_squared_error/num_present/zeros_like	ZerosLike&GraphRegularization/Reshape_1:output:0\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
MGraphRegularization/graph_loss/mean_squared_error/num_present/ones_like/ShapeShape&GraphRegularization/Reshape_1:output:0\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
MGraphRegularization/graph_loss/mean_squared_error/num_present/ones_like/ConstConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
GGraphRegularization/graph_loss/mean_squared_error/num_present/ones_likeFillVGraphRegularization/graph_loss/mean_squared_error/num_present/ones_like/Shape:output:0VGraphRegularization/graph_loss/mean_squared_error/num_present/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
DGraphRegularization/graph_loss/mean_squared_error/num_present/SelectSelectGGraphRegularization/graph_loss/mean_squared_error/num_present/Equal:z:0LGraphRegularization/graph_loss/mean_squared_error/num_present/zeros_like:y:0PGraphRegularization/graph_loss/mean_squared_error/num_present/ones_like:output:0*
T0*'
_output_shapes
:??????????
rGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeShapeMGraphRegularization/graph_loss/mean_squared_error/num_present/Select:output:0*
T0*
_output_shapes
:?
qGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
qGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShapeGGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:z:0\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
pGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
pGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
nGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualyGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/x:output:0zGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?

sGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shapeStatelessIfrGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0rGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0yGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:output:0zGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0zGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0{GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch?R?
~GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_5575*
output_shapes
: *?
then_branch?R
}GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_5574?
|GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity|GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
jGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
lGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
lGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*W
valueNBL BFGraphRegularization/graph_loss/mean_squared_error/num_present/Select:0?
lGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
lGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*V
valueMBK BEGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:0?
lGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?

pGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardIf?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0?GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0{GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0zGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0rGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0S^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branchR}
{GraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_5629*
output_shapes
: *?
then_branch~R|
zGraphRegularization_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_5628?
yGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentityyGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
_GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShapeGGraphRegularization/graph_loss/mean_squared_error/SquaredDifference:z:0\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityz^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
_GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityz^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
YGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_likeFillhGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Shape:output:0hGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
OGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weightsMulMGraphRegularization/graph_loss/mean_squared_error/num_present/Select:output:0bGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/ones_like:output:0*
T0*'
_output_shapes
:??????????
CGraphRegularization/graph_loss/mean_squared_error/num_present/ConstConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
=GraphRegularization/graph_loss/mean_squared_error/num_presentSumSGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights:z:0LGraphRegularization/graph_loss/mean_squared_error/num_present/Const:output:0*
T0*
_output_shapes
: ?
6GraphRegularization/graph_loss/mean_squared_error/RankConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
=GraphRegularization/graph_loss/mean_squared_error/range/startConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
=GraphRegularization/graph_loss/mean_squared_error/range/deltaConst\^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
7GraphRegularization/graph_loss/mean_squared_error/rangeRangeFGraphRegularization/graph_loss/mean_squared_error/range/start:output:0?GraphRegularization/graph_loss/mean_squared_error/Rank:output:0FGraphRegularization/graph_loss/mean_squared_error/range/delta:output:0*
_output_shapes
: ?
7GraphRegularization/graph_loss/mean_squared_error/Sum_1Sum>GraphRegularization/graph_loss/mean_squared_error/Sum:output:0@GraphRegularization/graph_loss/mean_squared_error/range:output:0*
T0*
_output_shapes
: ?
7GraphRegularization/graph_loss/mean_squared_error/valueDivNoNan@GraphRegularization/graph_loss/mean_squared_error/Sum_1:output:0FGraphRegularization/graph_loss/mean_squared_error/num_present:output:0*
T0*
_output_shapes
: k
&GraphRegularization/graph_loss/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@?
$GraphRegularization/graph_loss/mul_1Mul;GraphRegularization/graph_loss/mean_squared_error/value:z:0/GraphRegularization/graph_loss/mul_1/y:output:0*
T0*
_output_shapes
: ^
GraphRegularization/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
GraphRegularization/mulMul"GraphRegularization/mul/x:output:0(GraphRegularization/graph_loss/mul_1:z:0*
T0*
_output_shapes
: Z
GraphRegularization/RankConst*
_output_shapes
: *
dtype0*
value	B : a
GraphRegularization/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
GraphRegularization/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
GraphRegularization/rangeRange(GraphRegularization/range/start:output:0!GraphRegularization/Rank:output:0(GraphRegularization/range/delta:output:0*
_output_shapes
: ?
GraphRegularization/SumSumGraphRegularization/mul:z:0"GraphRegularization/range:output:0*
T0*
_output_shapes
: ?
'GraphRegularization/AssignAddVariableOpAssignAddVariableOp0graphregularization_assignaddvariableop_resource GraphRegularization/Sum:output:0*
_output_shapes
 *
dtype0Z
GraphRegularization/SizeConst*
_output_shapes
: *
dtype0*
value	B :s
GraphRegularization/CastCast!GraphRegularization/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
)GraphRegularization/AssignAddVariableOp_1AssignAddVariableOp2graphregularization_assignaddvariableop_1_resourceGraphRegularization/Cast:y:0(^GraphRegularization/AssignAddVariableOp*
_output_shapes
 *
dtype0?
-GraphRegularization/div_no_nan/ReadVariableOpReadVariableOp0graphregularization_assignaddvariableop_resource(^GraphRegularization/AssignAddVariableOp*^GraphRegularization/AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
/GraphRegularization/div_no_nan/ReadVariableOp_1ReadVariableOp2graphregularization_assignaddvariableop_1_resource*^GraphRegularization/AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
GraphRegularization/div_no_nanDivNoNan5GraphRegularization/div_no_nan/ReadVariableOp:value:07GraphRegularization/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: m
GraphRegularization/IdentityIdentity"GraphRegularization/div_no_nan:z:0*
T0*
_output_shapes
: ?
IdentityIdentity8GraphRegularization/sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^GraphRegularization/AssignAddVariableOp*^GraphRegularization/AssignAddVariableOp_1.^GraphRegularization/div_no_nan/ReadVariableOp0^GraphRegularization/div_no_nan/ReadVariableOp_1B^GraphRegularization/graph_loss/assert_greater_equal/Assert/AssertS^GraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuardq^GraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard=^GraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOp?^GraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOp<^GraphRegularization/sequential/conv2d/Conv2D/ReadVariableOp>^GraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOp?^GraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOpA^GraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOp>^GraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOp@^GraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOp?^GraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOpA^GraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOp>^GraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOp@^GraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOp<^GraphRegularization/sequential/dense/BiasAdd/ReadVariableOp>^GraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOp;^GraphRegularization/sequential/dense/MatMul/ReadVariableOp=^GraphRegularization/sequential/dense/MatMul_1/ReadVariableOp>^GraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOp@^GraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOp=^GraphRegularization/sequential/dense_1/MatMul/ReadVariableOp?^GraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2R
'GraphRegularization/AssignAddVariableOp'GraphRegularization/AssignAddVariableOp2V
)GraphRegularization/AssignAddVariableOp_1)GraphRegularization/AssignAddVariableOp_12^
-GraphRegularization/div_no_nan/ReadVariableOp-GraphRegularization/div_no_nan/ReadVariableOp2b
/GraphRegularization/div_no_nan/ReadVariableOp_1/GraphRegularization/div_no_nan/ReadVariableOp_12?
AGraphRegularization/graph_loss/assert_greater_equal/Assert/AssertAGraphRegularization/graph_loss/assert_greater_equal/Assert/Assert2?
RGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuardRGraphRegularization/graph_loss/mean_squared_error/assert_broadcastable/AssertGuard2?
pGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardpGraphRegularization/graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard2|
<GraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOp<GraphRegularization/sequential/conv2d/BiasAdd/ReadVariableOp2?
>GraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOp>GraphRegularization/sequential/conv2d/BiasAdd_1/ReadVariableOp2z
;GraphRegularization/sequential/conv2d/Conv2D/ReadVariableOp;GraphRegularization/sequential/conv2d/Conv2D/ReadVariableOp2~
=GraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOp=GraphRegularization/sequential/conv2d/Conv2D_1/ReadVariableOp2?
>GraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOp>GraphRegularization/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
@GraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOp@GraphRegularization/sequential/conv2d_1/BiasAdd_1/ReadVariableOp2~
=GraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOp=GraphRegularization/sequential/conv2d_1/Conv2D/ReadVariableOp2?
?GraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOp?GraphRegularization/sequential/conv2d_1/Conv2D_1/ReadVariableOp2?
>GraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOp>GraphRegularization/sequential/conv2d_2/BiasAdd/ReadVariableOp2?
@GraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOp@GraphRegularization/sequential/conv2d_2/BiasAdd_1/ReadVariableOp2~
=GraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOp=GraphRegularization/sequential/conv2d_2/Conv2D/ReadVariableOp2?
?GraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOp?GraphRegularization/sequential/conv2d_2/Conv2D_1/ReadVariableOp2z
;GraphRegularization/sequential/dense/BiasAdd/ReadVariableOp;GraphRegularization/sequential/dense/BiasAdd/ReadVariableOp2~
=GraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOp=GraphRegularization/sequential/dense/BiasAdd_1/ReadVariableOp2x
:GraphRegularization/sequential/dense/MatMul/ReadVariableOp:GraphRegularization/sequential/dense/MatMul/ReadVariableOp2|
<GraphRegularization/sequential/dense/MatMul_1/ReadVariableOp<GraphRegularization/sequential/dense/MatMul_1/ReadVariableOp2~
=GraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOp=GraphRegularization/sequential/dense_1/BiasAdd/ReadVariableOp2?
?GraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOp?GraphRegularization/sequential/dense_1/BiasAdd_1/ReadVariableOp2|
<GraphRegularization/sequential/dense_1/MatMul/ReadVariableOp<GraphRegularization/sequential/dense_1/MatMul/ReadVariableOp2?
>GraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOp>GraphRegularization/sequential/dense_1/MatMul_1/ReadVariableOp:q m
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_0_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_0_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_1_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_1_weight:qm
A
_output_shapes/
-:+???????????????????????????
(
_user_specified_nameNL_nbr_2_image:XT
'
_output_shapes
:?????????
)
_user_specified_nameNL_nbr_2_weight:XT
1
_output_shapes
:???????????

_user_specified_nameimage
?

?
?__inference_dense_layer_call_and_return_conditional_losses_5813

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_5800

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&& :W S
/
_output_shapes
:?????????&& 
 
_user_specified_nameinputs
??
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7754
inputs_nl_nbr_0_image
inputs_nl_nbr_0_weight
inputs_nl_nbr_1_image
inputs_nl_nbr_1_weight
inputs_nl_nbr_2_image
inputs_nl_nbr_2_weight
inputs_imageJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:L
2sequential_conv2d_2_conv2d_readvariableop_resource: A
3sequential_conv2d_2_biasadd_readvariableop_resource: C
/sequential_dense_matmul_readvariableop_resource:
??@>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@
2sequential_dense_1_biasadd_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?-graph_loss/assert_greater_equal/Assert/Assert?>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard?\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard?(sequential/conv2d/BiasAdd/ReadVariableOp?*sequential/conv2d/BiasAdd_1/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?)sequential/conv2d/Conv2D_1/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?,sequential/conv2d_1/BiasAdd_1/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?+sequential/conv2d_1/Conv2D_1/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?,sequential/conv2d_2/BiasAdd_1/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?+sequential/conv2d_2/Conv2D_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/BiasAdd_1/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?(sequential/dense/MatMul_1/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/BiasAdd_1/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?*sequential/dense_1/MatMul_1/ReadVariableOpA
ShapeShapeinputs_image*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPackinputs_nl_nbr_0_imageinputs_nl_nbr_1_imageinputs_nl_nbr_2_image*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
stack_1Packinputs_nl_nbr_0_weightinputs_nl_nbr_1_weightinputs_nl_nbr_2_weight*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:?????????p
sequential/CastCastinputs_image*

DstT0*

SrcT0*1
_output_shapes
:????????????
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d/Conv2DConv2Dsequential/Cast:y:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????~
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
"sequential/max_pooling2d_2/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????v
sequential/Cast_1CastReshape:output:0*

DstT0*

SrcT0*1
_output_shapes
:????????????
)sequential/conv2d/Conv2D_1/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d/Conv2D_1Conv2Dsequential/Cast_1:y:01sequential/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*sequential/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d/BiasAdd_1BiasAdd#sequential/conv2d/Conv2D_1:output:02sequential/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d/Relu_1Relu$sequential/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
"sequential/max_pooling2d/MaxPool_1MaxPool&sequential/conv2d/Relu_1:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
+sequential/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d_1/Conv2D_1Conv2D+sequential/max_pooling2d/MaxPool_1:output:03sequential/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
,sequential/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d_1/BiasAdd_1BiasAdd%sequential/conv2d_1/Conv2D_1:output:04sequential/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d_1/Relu_1Relu&sequential/conv2d_1/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
$sequential/max_pooling2d_1/MaxPool_1MaxPool(sequential/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
+sequential/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential/conv2d_2/Conv2D_1Conv2D-sequential/max_pooling2d_1/MaxPool_1:output:03sequential/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
,sequential/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential/conv2d_2/BiasAdd_1BiasAdd%sequential/conv2d_2/Conv2D_1:output:04sequential/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
sequential/conv2d_2/Relu_1Relu&sequential/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????KK ?
$sequential/max_pooling2d_2/MaxPool_1MaxPool(sequential/conv2d_2/Relu_1:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
k
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"??????  ?
sequential/flatten/Reshape_1Reshape-sequential/max_pooling2d_2/MaxPool_1:output:0#sequential/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????@?
*sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense_1/MatMul_1MatMul%sequential/dense/Relu_1:activations:02sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAdd_1BiasAdd%sequential/dense_1/MatMul_1:product:03sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential/dense_1/Softmax_1Softmax%sequential/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????f
graph_loss/ShapeShape&sequential/dense_1/Softmax_1:softmax:0*
T0*
_output_shapes
:h
graph_loss/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 graph_loss/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 graph_loss/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_sliceStridedSlicegraph_loss/Shape:output:0'graph_loss/strided_slice/stack:output:0)graph_loss/strided_slice/stack_1:output:0)graph_loss/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
graph_loss/Shape_1Shape$sequential/dense_1/Softmax:softmax:0*
T0*
_output_shapes
:j
 graph_loss/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"graph_loss/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"graph_loss/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_slice_1StridedSlicegraph_loss/Shape_1:output:0)graph_loss/strided_slice_1/stack:output:0+graph_loss/strided_slice_1/stack_1:output:0+graph_loss/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
graph_loss/floordivFloorDiv!graph_loss/strided_slice:output:0#graph_loss/strided_slice_1:output:0*
T0*
_output_shapes
: c
!graph_loss/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B : ?
,graph_loss/assert_greater_equal/GreaterEqualGreaterEqualgraph_loss/floordiv:z:0*graph_loss/assert_greater_equal/y:output:0*
T0*
_output_shapes
: f
$graph_loss/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : m
+graph_loss/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+graph_loss/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%graph_loss/assert_greater_equal/rangeRange4graph_loss/assert_greater_equal/range/start:output:0-graph_loss/assert_greater_equal/Rank:output:04graph_loss/assert_greater_equal/range/delta:output:0*
_output_shapes
: ?
#graph_loss/assert_greater_equal/AllAll0graph_loss/assert_greater_equal/GreaterEqual:z:0.graph_loss/assert_greater_equal/range:output:0*
_output_shapes
: ?
,graph_loss/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
.graph_loss/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*-
value$B" Bx (graph_loss/floordiv:0) = ?
.graph_loss/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (graph_loss/assert_greater_equal/y:0) = ?
4graph_loss/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
4graph_loss/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*-
value$B" Bx (graph_loss/floordiv:0) = ?
4graph_loss/assert_greater_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (graph_loss/assert_greater_equal/y:0) = ?
-graph_loss/assert_greater_equal/Assert/AssertAssert,graph_loss/assert_greater_equal/All:output:0=graph_loss/assert_greater_equal/Assert/Assert/data_0:output:0=graph_loss/assert_greater_equal/Assert/Assert/data_1:output:0graph_loss/floordiv:z:0=graph_loss/assert_greater_equal/Assert/Assert/data_3:output:0*graph_loss/assert_greater_equal/y:output:0*
T	
2*
_output_shapes
 ?
graph_loss/Shape_2Shape$sequential/dense_1/Softmax:softmax:0.^graph_loss/assert_greater_equal/Assert/Assert*
T0*
_output_shapes
:?
 graph_loss/strided_slice_2/stackConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"graph_loss/strided_slice_2/stack_1Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
"graph_loss/strided_slice_2/stack_2Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_slice_2StridedSlicegraph_loss/Shape_2:output:0)graph_loss/strided_slice_2/stack:output:0+graph_loss/strided_slice_2/stack_1:output:0+graph_loss/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
graph_loss/range/startConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
graph_loss/range/deltaConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
graph_loss/rangeRangegraph_loss/range/start:output:0#graph_loss/strided_slice_2:output:0graph_loss/range/delta:output:0*#
_output_shapes
:??????????
graph_loss/ExpandDims/dimConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
valueB :
??????????
graph_loss/ExpandDims
ExpandDimsgraph_loss/range:output:0"graph_loss/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
graph_loss/Tile/multiples/0Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
graph_loss/Tile/multiplesPack$graph_loss/Tile/multiples/0:output:0graph_loss/floordiv:z:0*
N*
T0*
_output_shapes
:?
graph_loss/TileTilegraph_loss/ExpandDims:output:0"graph_loss/Tile/multiples:output:0*
T0*0
_output_shapes
:??????????????????t
graph_loss/mulMul#graph_loss/strided_slice_2:output:0graph_loss/floordiv:z:0*
T0*
_output_shapes
: b
graph_loss/Reshape/shapePackgraph_loss/mul:z:0*
N*
T0*
_output_shapes
:?
graph_loss/ReshapeReshapegraph_loss/Tile:output:0!graph_loss/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
graph_loss/GatherV2/axisConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
graph_loss/GatherV2GatherV2$sequential/dense_1/Softmax:softmax:0graph_loss/Reshape:output:0!graph_loss/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:??????????
/graph_loss/mean_squared_error/SquaredDifferenceSquaredDifference&sequential/dense_1/Softmax_1:softmax:0graph_loss/GatherV2:output:0*
T0*'
_output_shapes
:??????????
@graph_loss/mean_squared_error/assert_broadcastable/weights/shapeShapeReshape_1:output:0*
T0*
_output_shapes
:?
?graph_loss/mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/assert_broadcastable/values/shapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:?
>graph_loss/mean_squared_error/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :?
>graph_loss/mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
<graph_loss/mean_squared_error/assert_broadcastable/is_scalarEqualGgraph_loss/mean_squared_error/assert_broadcastable/is_scalar/x:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
Agraph_loss/mean_squared_error/assert_broadcastable/is_valid_shapeStatelessIf@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0Ggraph_loss/mean_squared_error/assert_broadcastable/values/rank:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0Igraph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *_
else_branchPRN
Lgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_7552*
output_shapes
: *^
then_branchORM
Kgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_7551?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentityJgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
8graph_loss/mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
:graph_loss/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
:graph_loss/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*
valueB BReshape_1:0?
:graph_loss/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
:graph_loss/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
:graph_loss/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
>graph_loss/mean_squared_error/assert_broadcastable/AssertGuardIfSgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Sgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Igraph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0.^graph_loss/assert_greater_equal/Assert/Assert*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *\
else_branchMRK
Igraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_7606*
output_shapes
: *[
then_branchLRJ
Hgraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_7605?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentityGgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
!graph_loss/mean_squared_error/MulMul3graph_loss/mean_squared_error/SquaredDifference:z:0Reshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
#graph_loss/mean_squared_error/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
!graph_loss/mean_squared_error/SumSum%graph_loss/mean_squared_error/Mul:z:0,graph_loss/mean_squared_error/Const:output:0*
T0*
_output_shapes
: ?
1graph_loss/mean_squared_error/num_present/Equal/yConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    ?
/graph_loss/mean_squared_error/num_present/EqualEqualReshape_1:output:0:graph_loss/mean_squared_error/num_present/Equal/y:output:0*
T0*'
_output_shapes
:??????????
4graph_loss/mean_squared_error/num_present/zeros_like	ZerosLikeReshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
9graph_loss/mean_squared_error/num_present/ones_like/ShapeShapeReshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
9graph_loss/mean_squared_error/num_present/ones_like/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
3graph_loss/mean_squared_error/num_present/ones_likeFillBgraph_loss/mean_squared_error/num_present/ones_like/Shape:output:0Bgraph_loss/mean_squared_error/num_present/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
0graph_loss/mean_squared_error/num_present/SelectSelect3graph_loss/mean_squared_error/num_present/Equal:z:08graph_loss/mean_squared_error/num_present/zeros_like:y:0<graph_loss/mean_squared_error/num_present/ones_like:output:0*
T0*'
_output_shapes
:??????????
^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeShape9graph_loss/mean_squared_error/num_present/Select:output:0*
T0*
_output_shapes
:?
]graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
]graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
Zgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualegraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/x:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
_graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shapeStatelessIf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *}
else_branchnRl
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_7646*
output_shapes
: *|
then_branchmRk
igraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_7645?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentityhgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
Vgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*C
value:B8 B2graph_loss/mean_squared_error/num_present/Select:0?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardIfqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0qgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0?^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *z
else_branchkRi
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_7700*
output_shapes
: *y
then_branchjRh
fgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_7699?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentityegraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
Kgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
Kgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
Egraph_loss/mean_squared_error/num_present/broadcast_weights/ones_likeFillTgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Shape:output:0Tgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
;graph_loss/mean_squared_error/num_present/broadcast_weightsMul9graph_loss/mean_squared_error/num_present/Select:output:0Ngraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like:output:0*
T0*'
_output_shapes
:??????????
/graph_loss/mean_squared_error/num_present/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
)graph_loss/mean_squared_error/num_presentSum?graph_loss/mean_squared_error/num_present/broadcast_weights:z:08graph_loss/mean_squared_error/num_present/Const:output:0*
T0*
_output_shapes
: ?
"graph_loss/mean_squared_error/RankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
)graph_loss/mean_squared_error/range/startConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
)graph_loss/mean_squared_error/range/deltaConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
#graph_loss/mean_squared_error/rangeRange2graph_loss/mean_squared_error/range/start:output:0+graph_loss/mean_squared_error/Rank:output:02graph_loss/mean_squared_error/range/delta:output:0*
_output_shapes
: ?
#graph_loss/mean_squared_error/Sum_1Sum*graph_loss/mean_squared_error/Sum:output:0,graph_loss/mean_squared_error/range:output:0*
T0*
_output_shapes
: ?
#graph_loss/mean_squared_error/valueDivNoNan,graph_loss/mean_squared_error/Sum_1:output:02graph_loss/mean_squared_error/num_present:output:0*
T0*
_output_shapes
: W
graph_loss/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@~
graph_loss/mul_1Mul'graph_loss/mean_squared_error/value:z:0graph_loss/mul_1/y:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>Q
mulMulmul/x:output:0graph_loss/mul_1:z:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: u

Identity_1Identity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?	
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1.^graph_loss/assert_greater_equal/Assert/Assert?^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard]^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard)^sequential/conv2d/BiasAdd/ReadVariableOp+^sequential/conv2d/BiasAdd_1/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp*^sequential/conv2d/Conv2D_1/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp-^sequential/conv2d_1/BiasAdd_1/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp,^sequential/conv2d_1/Conv2D_1/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp-^sequential/conv2d_2/BiasAdd_1/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp,^sequential/conv2d_2/Conv2D_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/BiasAdd_1/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp+^sequential/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12^
-graph_loss/assert_greater_equal/Assert/Assert-graph_loss/assert_greater_equal/Assert/Assert2?
>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard2?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2X
*sequential/conv2d/BiasAdd_1/ReadVariableOp*sequential/conv2d/BiasAdd_1/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2V
)sequential/conv2d/Conv2D_1/ReadVariableOp)sequential/conv2d/Conv2D_1/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2\
,sequential/conv2d_1/BiasAdd_1/ReadVariableOp,sequential/conv2d_1/BiasAdd_1/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2Z
+sequential/conv2d_1/Conv2D_1/ReadVariableOp+sequential/conv2d_1/Conv2D_1/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2\
,sequential/conv2d_2/BiasAdd_1/ReadVariableOp,sequential/conv2d_2/BiasAdd_1/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2Z
+sequential/conv2d_2/Conv2D_1/ReadVariableOp+sequential/conv2d_2/Conv2D_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/BiasAdd_1/ReadVariableOp+sequential/dense_1/BiasAdd_1/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2X
*sequential/dense_1/MatMul_1/ReadVariableOp*sequential/dense_1/MatMul_1/ReadVariableOp:x t
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_0_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_0_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_1_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_1_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_2_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_2_weight:_[
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?

?
fgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_7561k
ggraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholderm
igraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder_1?
?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank
h
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
dgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity_graph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank*
T0
*
_output_shapes
: "?
dgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identitymgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?6
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_7305?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
?graph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
??
?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7405
inputs_nl_nbr_0_image
inputs_nl_nbr_0_weight
inputs_nl_nbr_1_image
inputs_nl_nbr_1_weight
inputs_nl_nbr_2_image
inputs_nl_nbr_2_weight
inputs_imageJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:L
2sequential_conv2d_2_conv2d_readvariableop_resource: A
3sequential_conv2d_2_biasadd_readvariableop_resource: C
/sequential_dense_matmul_readvariableop_resource:
??@>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@
2sequential_dense_1_biasadd_readvariableop_resource:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?-graph_loss/assert_greater_equal/Assert/Assert?>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard?\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard?(sequential/conv2d/BiasAdd/ReadVariableOp?*sequential/conv2d/BiasAdd_1/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?)sequential/conv2d/Conv2D_1/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?,sequential/conv2d_1/BiasAdd_1/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?+sequential/conv2d_1/Conv2D_1/ReadVariableOp?*sequential/conv2d_2/BiasAdd/ReadVariableOp?,sequential/conv2d_2/BiasAdd_1/ReadVariableOp?)sequential/conv2d_2/Conv2D/ReadVariableOp?+sequential/conv2d_2/Conv2D_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/BiasAdd_1/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?(sequential/dense/MatMul_1/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/BiasAdd_1/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?*sequential/dense_1/MatMul_1/ReadVariableOpA
ShapeShapeinputs_image*
T0*
_output_shapes
:X
ConstConst*
_output_shapes
:*
dtype0*
valueB:
?????????]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Const:output:0strided_slice:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:?
stackPackinputs_nl_nbr_0_imageinputs_nl_nbr_1_imageinputs_nl_nbr_2_image*
N*
T0*E
_output_shapes3
1:/???????????????????????????*

axiso
ReshapeReshapestack:output:0concat:output:0*
T0*1
_output_shapes
:???????????Z
Const_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????[
concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2Const_1:output:0concat_1/values_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
stack_1Packinputs_nl_nbr_0_weightinputs_nl_nbr_1_weightinputs_nl_nbr_2_weight*
N*
T0*+
_output_shapes
:?????????*

axisk
	Reshape_1Reshapestack_1:output:0concat_1:output:0*
T0*'
_output_shapes
:?????????p
sequential/CastCastinputs_image*

DstT0*

SrcT0*1
_output_shapes
:????????????
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d/Conv2DConv2Dsequential/Cast:y:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????~
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:????????????
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????KK ?
"sequential/max_pooling2d_2/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"??????  ?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????v
sequential/Cast_1CastReshape:output:0*

DstT0*

SrcT0*1
_output_shapes
:????????????
)sequential/conv2d/Conv2D_1/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d/Conv2D_1Conv2Dsequential/Cast_1:y:01sequential/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*sequential/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d/BiasAdd_1BiasAdd#sequential/conv2d/Conv2D_1:output:02sequential/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d/Relu_1Relu$sequential/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
"sequential/max_pooling2d/MaxPool_1MaxPool&sequential/conv2d/Relu_1:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
?
+sequential/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential/conv2d_1/Conv2D_1Conv2D+sequential/max_pooling2d/MaxPool_1:output:03sequential/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
,sequential/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv2d_1/BiasAdd_1BiasAdd%sequential/conv2d_1/Conv2D_1:output:04sequential/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
sequential/conv2d_1/Relu_1Relu&sequential/conv2d_1/BiasAdd_1:output:0*
T0*1
_output_shapes
:????????????
$sequential/max_pooling2d_1/MaxPool_1MaxPool(sequential/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????KK*
ksize
*
paddingSAME*
strides
?
+sequential/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential/conv2d_2/Conv2D_1Conv2D-sequential/max_pooling2d_1/MaxPool_1:output:03sequential/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK *
paddingSAME*
strides
?
,sequential/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential/conv2d_2/BiasAdd_1BiasAdd%sequential/conv2d_2/Conv2D_1:output:04sequential/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????KK ?
sequential/conv2d_2/Relu_1Relu&sequential/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????KK ?
$sequential/max_pooling2d_2/MaxPool_1MaxPool(sequential/conv2d_2/Relu_1:activations:0*/
_output_shapes
:?????????&& *
ksize
*
paddingSAME*
strides
k
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"??????  ?
sequential/flatten/Reshape_1Reshape-sequential/max_pooling2d_2/MaxPool_1:output:0#sequential/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????@?
*sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense_1/MatMul_1MatMul%sequential/dense/Relu_1:activations:02sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAdd_1BiasAdd%sequential/dense_1/MatMul_1:product:03sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential/dense_1/Softmax_1Softmax%sequential/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????f
graph_loss/ShapeShape&sequential/dense_1/Softmax_1:softmax:0*
T0*
_output_shapes
:h
graph_loss/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 graph_loss/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 graph_loss/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_sliceStridedSlicegraph_loss/Shape:output:0'graph_loss/strided_slice/stack:output:0)graph_loss/strided_slice/stack_1:output:0)graph_loss/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
graph_loss/Shape_1Shape$sequential/dense_1/Softmax:softmax:0*
T0*
_output_shapes
:j
 graph_loss/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"graph_loss/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"graph_loss/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_slice_1StridedSlicegraph_loss/Shape_1:output:0)graph_loss/strided_slice_1/stack:output:0+graph_loss/strided_slice_1/stack_1:output:0+graph_loss/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
graph_loss/floordivFloorDiv!graph_loss/strided_slice:output:0#graph_loss/strided_slice_1:output:0*
T0*
_output_shapes
: c
!graph_loss/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B : ?
,graph_loss/assert_greater_equal/GreaterEqualGreaterEqualgraph_loss/floordiv:z:0*graph_loss/assert_greater_equal/y:output:0*
T0*
_output_shapes
: f
$graph_loss/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : m
+graph_loss/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+graph_loss/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%graph_loss/assert_greater_equal/rangeRange4graph_loss/assert_greater_equal/range/start:output:0-graph_loss/assert_greater_equal/Rank:output:04graph_loss/assert_greater_equal/range/delta:output:0*
_output_shapes
: ?
#graph_loss/assert_greater_equal/AllAll0graph_loss/assert_greater_equal/GreaterEqual:z:0.graph_loss/assert_greater_equal/range:output:0*
_output_shapes
: ?
,graph_loss/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
.graph_loss/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*-
value$B" Bx (graph_loss/floordiv:0) = ?
.graph_loss/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (graph_loss/assert_greater_equal/y:0) = ?
4graph_loss/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:?
4graph_loss/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*-
value$B" Bx (graph_loss/floordiv:0) = ?
4graph_loss/assert_greater_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (graph_loss/assert_greater_equal/y:0) = ?
-graph_loss/assert_greater_equal/Assert/AssertAssert,graph_loss/assert_greater_equal/All:output:0=graph_loss/assert_greater_equal/Assert/Assert/data_0:output:0=graph_loss/assert_greater_equal/Assert/Assert/data_1:output:0graph_loss/floordiv:z:0=graph_loss/assert_greater_equal/Assert/Assert/data_3:output:0*graph_loss/assert_greater_equal/y:output:0*
T	
2*
_output_shapes
 ?
graph_loss/Shape_2Shape$sequential/dense_1/Softmax:softmax:0.^graph_loss/assert_greater_equal/Assert/Assert*
T0*
_output_shapes
:?
 graph_loss/strided_slice_2/stackConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"graph_loss/strided_slice_2/stack_1Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
"graph_loss/strided_slice_2/stack_2Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:?
graph_loss/strided_slice_2StridedSlicegraph_loss/Shape_2:output:0)graph_loss/strided_slice_2/stack:output:0+graph_loss/strided_slice_2/stack_1:output:0+graph_loss/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
graph_loss/range/startConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
graph_loss/range/deltaConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
graph_loss/rangeRangegraph_loss/range/start:output:0#graph_loss/strided_slice_2:output:0graph_loss/range/delta:output:0*#
_output_shapes
:??????????
graph_loss/ExpandDims/dimConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
valueB :
??????????
graph_loss/ExpandDims
ExpandDimsgraph_loss/range:output:0"graph_loss/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
graph_loss/Tile/multiples/0Const.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B :?
graph_loss/Tile/multiplesPack$graph_loss/Tile/multiples/0:output:0graph_loss/floordiv:z:0*
N*
T0*
_output_shapes
:?
graph_loss/TileTilegraph_loss/ExpandDims:output:0"graph_loss/Tile/multiples:output:0*
T0*0
_output_shapes
:??????????????????t
graph_loss/mulMul#graph_loss/strided_slice_2:output:0graph_loss/floordiv:z:0*
T0*
_output_shapes
: b
graph_loss/Reshape/shapePackgraph_loss/mul:z:0*
N*
T0*
_output_shapes
:?
graph_loss/ReshapeReshapegraph_loss/Tile:output:0!graph_loss/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
graph_loss/GatherV2/axisConst.^graph_loss/assert_greater_equal/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
graph_loss/GatherV2GatherV2$sequential/dense_1/Softmax:softmax:0graph_loss/Reshape:output:0!graph_loss/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:??????????
/graph_loss/mean_squared_error/SquaredDifferenceSquaredDifference&sequential/dense_1/Softmax_1:softmax:0graph_loss/GatherV2:output:0*
T0*'
_output_shapes
:??????????
@graph_loss/mean_squared_error/assert_broadcastable/weights/shapeShapeReshape_1:output:0*
T0*
_output_shapes
:?
?graph_loss/mean_squared_error/assert_broadcastable/weights/rankConst*
_output_shapes
: *
dtype0*
value	B :?
?graph_loss/mean_squared_error/assert_broadcastable/values/shapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:?
>graph_loss/mean_squared_error/assert_broadcastable/values/rankConst*
_output_shapes
: *
dtype0*
value	B :?
>graph_loss/mean_squared_error/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
dtype0*
value	B : ?
<graph_loss/mean_squared_error/assert_broadcastable/is_scalarEqualGgraph_loss/mean_squared_error/assert_broadcastable/is_scalar/x:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
Agraph_loss/mean_squared_error/assert_broadcastable/is_valid_shapeStatelessIf@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0Ggraph_loss/mean_squared_error/assert_broadcastable/values/rank:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/weights/rank:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0Igraph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *_
else_branchPRN
Lgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_false_7203*
output_shapes
: *^
then_branchORM
Kgraph_loss_mean_squared_error_assert_broadcastable_is_valid_shape_true_7202?
Jgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/IdentityIdentityJgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
8graph_loss/mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
:graph_loss/mean_squared_error/assert_broadcastable/Const_1Const*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
:graph_loss/mean_squared_error/assert_broadcastable/Const_2Const*
_output_shapes
: *
dtype0*
valueB BReshape_1:0?
:graph_loss/mean_squared_error/assert_broadcastable/Const_3Const*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
:graph_loss/mean_squared_error/assert_broadcastable/Const_4Const*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
:graph_loss/mean_squared_error/assert_broadcastable/Const_5Const*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
>graph_loss/mean_squared_error/assert_broadcastable/AssertGuardIfSgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Sgraph_loss/mean_squared_error/assert_broadcastable/is_valid_shape/Identity:output:0Igraph_loss/mean_squared_error/assert_broadcastable/weights/shape:output:0Hgraph_loss/mean_squared_error/assert_broadcastable/values/shape:output:0@graph_loss/mean_squared_error/assert_broadcastable/is_scalar:z:0.^graph_loss/assert_greater_equal/Assert/Assert*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *\
else_branchMRK
Igraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_false_7257*
output_shapes
: *[
then_branchLRJ
Hgraph_loss_mean_squared_error_assert_broadcastable_AssertGuard_true_7256?
Ggraph_loss/mean_squared_error/assert_broadcastable/AssertGuard/IdentityIdentityGgraph_loss/mean_squared_error/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
!graph_loss/mean_squared_error/MulMul3graph_loss/mean_squared_error/SquaredDifference:z:0Reshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
#graph_loss/mean_squared_error/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
!graph_loss/mean_squared_error/SumSum%graph_loss/mean_squared_error/Mul:z:0,graph_loss/mean_squared_error/Const:output:0*
T0*
_output_shapes
: ?
1graph_loss/mean_squared_error/num_present/Equal/yConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *    ?
/graph_loss/mean_squared_error/num_present/EqualEqualReshape_1:output:0:graph_loss/mean_squared_error/num_present/Equal/y:output:0*
T0*'
_output_shapes
:??????????
4graph_loss/mean_squared_error/num_present/zeros_like	ZerosLikeReshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*'
_output_shapes
:??????????
9graph_loss/mean_squared_error/num_present/ones_like/ShapeShapeReshape_1:output:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
9graph_loss/mean_squared_error/num_present/ones_like/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
3graph_loss/mean_squared_error/num_present/ones_likeFillBgraph_loss/mean_squared_error/num_present/ones_like/Shape:output:0Bgraph_loss/mean_squared_error/num_present/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
0graph_loss/mean_squared_error/num_present/SelectSelect3graph_loss/mean_squared_error/num_present/Equal:z:08graph_loss/mean_squared_error/num_present/zeros_like:y:0<graph_loss/mean_squared_error/num_present/ones_like:output:0*
T0*'
_output_shapes
:??????????
^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeShape9graph_loss/mean_squared_error/num_present/Select:output:0*
T0*
_output_shapes
:?
]graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
]graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
Zgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualegraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/x:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0*
T0*
_output_shapes
: ?
_graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shapeStatelessIf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *}
else_branchnRl
jgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_7297*
output_shapes
: *|
then_branchmRk
igraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_true_7296?
hgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentityhgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape:output:0*
T0
*
_output_shapes
: ?
Vgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*8
value/B- B'weights can not be broadcast to values.?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bweights.shape=?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*C
value:B8 B2graph_loss/mean_squared_error/num_present/Select:0?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB Bvalues.shape=?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*B
value9B7 B1graph_loss/mean_squared_error/SquaredDifference:0?
Xgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5ConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB B
is_scalar=?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuardIfqgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0qgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0ggraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:output:0fgraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:output:0^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:z:0?^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *z
else_branchkRi
ggraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_false_7351*
output_shapes
: *y
then_branchjRh
fgraph_loss_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_AssertGuard_true_7350?
egraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/IdentityIdentityegraph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard:output:0*
T0
*
_output_shapes
: ?
Kgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape3graph_loss/mean_squared_error/SquaredDifference:z:0H^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
T0*
_output_shapes
:?
Kgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identityf^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ???
Egraph_loss/mean_squared_error/num_present/broadcast_weights/ones_likeFillTgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Shape:output:0Tgraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like/Const:output:0*
T0*'
_output_shapes
:??????????
;graph_loss/mean_squared_error/num_present/broadcast_weightsMul9graph_loss/mean_squared_error/num_present/Select:output:0Ngraph_loss/mean_squared_error/num_present/broadcast_weights/ones_like:output:0*
T0*'
_output_shapes
:??????????
/graph_loss/mean_squared_error/num_present/ConstConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       ?
)graph_loss/mean_squared_error/num_presentSum?graph_loss/mean_squared_error/num_present/broadcast_weights:z:08graph_loss/mean_squared_error/num_present/Const:output:0*
T0*
_output_shapes
: ?
"graph_loss/mean_squared_error/RankConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
)graph_loss/mean_squared_error/range/startConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ?
)graph_loss/mean_squared_error/range/deltaConstH^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :?
#graph_loss/mean_squared_error/rangeRange2graph_loss/mean_squared_error/range/start:output:0+graph_loss/mean_squared_error/Rank:output:02graph_loss/mean_squared_error/range/delta:output:0*
_output_shapes
: ?
#graph_loss/mean_squared_error/Sum_1Sum*graph_loss/mean_squared_error/Sum:output:0,graph_loss/mean_squared_error/range:output:0*
T0*
_output_shapes
: ?
#graph_loss/mean_squared_error/valueDivNoNan,graph_loss/mean_squared_error/Sum_1:output:02graph_loss/mean_squared_error/num_present:output:0*
T0*
_output_shapes
: W
graph_loss/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@~
graph_loss/mul_1Mul'graph_loss/mean_squared_error/value:z:0graph_loss/mul_1/y:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??L>Q
mulMulmul/x:output:0graph_loss/mul_1:z:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: D
SumSummul:z:0range:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: u

Identity_1Identity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????G

Identity_2Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?	
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1.^graph_loss/assert_greater_equal/Assert/Assert?^graph_loss/mean_squared_error/assert_broadcastable/AssertGuard]^graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard)^sequential/conv2d/BiasAdd/ReadVariableOp+^sequential/conv2d/BiasAdd_1/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp*^sequential/conv2d/Conv2D_1/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp-^sequential/conv2d_1/BiasAdd_1/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp,^sequential/conv2d_1/Conv2D_1/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp-^sequential/conv2d_2/BiasAdd_1/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp,^sequential/conv2d_2/Conv2D_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/BiasAdd_1/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp+^sequential/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:?????????:+???????????????????????????:?????????:+???????????????????????????:?????????:???????????: : : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12^
-graph_loss/assert_greater_equal/Assert/Assert-graph_loss/assert_greater_equal/Assert/Assert2?
>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard>graph_loss/mean_squared_error/assert_broadcastable/AssertGuard2?
\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard\graph_loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2X
*sequential/conv2d/BiasAdd_1/ReadVariableOp*sequential/conv2d/BiasAdd_1/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2V
)sequential/conv2d/Conv2D_1/ReadVariableOp)sequential/conv2d/Conv2D_1/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2\
,sequential/conv2d_1/BiasAdd_1/ReadVariableOp,sequential/conv2d_1/BiasAdd_1/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2Z
+sequential/conv2d_1/Conv2D_1/ReadVariableOp+sequential/conv2d_1/Conv2D_1/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2\
,sequential/conv2d_2/BiasAdd_1/ReadVariableOp,sequential/conv2d_2/BiasAdd_1/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2Z
+sequential/conv2d_2/Conv2D_1/ReadVariableOp+sequential/conv2d_2/Conv2D_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/BiasAdd_1/ReadVariableOp+sequential/dense_1/BiasAdd_1/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2X
*sequential/dense_1/MatMul_1/ReadVariableOp*sequential/dense_1/MatMul_1/ReadVariableOp:x t
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_0_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_0_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_1_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_1_weight:xt
A
_output_shapes/
-:+???????????????????????????
/
_user_specified_nameinputs/NL_nbr_2_image:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/NL_nbr_2_weight:_[
1
_output_shapes
:???????????
&
_user_specified_nameinputs/image
?3
?
xmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_8196?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape~
zmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_placeholder
{
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity
?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDims?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFill?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:output:0*
N*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDims?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_has_invalid_dims_expanddims_1_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shape?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperation?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:output:0*
T0*<
_output_shapes*
(:?????????:?????????:*
set_operationa-b?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:result_values:0*
T0*
_output_shapes
: ?
?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst*
_output_shapes
: *
dtype0*
value	B : ?
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqual?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:output:0?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:output:0*
T0*
_output_shapes
: ?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:z:0*
T0
*
_output_shapes
: "?
wmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_identity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::: :  

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_false_8188d
`mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_placeholder
?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?
?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapea
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identity
?
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqual?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_rank?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_is_same_rank_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_rank*
T0*
_output_shapes
: ?	
nmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shapeStatelessIfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_values_shape?mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_mean_squared_error_num_present_broadcast_weights_assert_broadcastable_weights_shapemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch}R{
ymean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_false_8197*
output_shapes
: *?
then_branch|Rz
xmean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_has_valid_nonscalar_shape_true_8196?
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/IdentityIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape:output:0*
T0
*
_output_shapes
: ?
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/IdentityIdentity?mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Identity:output:0*
T0
*
_output_shapes
: "?
]mean_squared_error_num_present_broadcast_weights_assert_broadcastable_is_valid_shape_identityfmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8340

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8395

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
c
NL_nbr_0_imageQ
 serving_default_NL_nbr_0_image:0+???????????????????????????
K
NL_nbr_0_weight8
!serving_default_NL_nbr_0_weight:0?????????
c
NL_nbr_1_imageQ
 serving_default_NL_nbr_1_image:0+???????????????????????????
K
NL_nbr_1_weight8
!serving_default_NL_nbr_1_weight:0?????????
c
NL_nbr_2_imageQ
 serving_default_NL_nbr_2_image:0+???????????????????????????
K
NL_nbr_2_weight8
!serving_default_NL_nbr_2_weight:0?????????
A
image8
serving_default_image:0???????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?

base_model
nbr_features_layer
regularizer
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?

layer_with_weights-0

layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
layer_with_weights-4
layer-8
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
(
	keras_api"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?"
	optimizer
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

"kernel
#bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
):' 2conv2d_2/kernel
: 2conv2d_2/bias
 :
??@2dense/kernel
:@2
dense/bias
 :@2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
 "
trackable_list_wrapper
7
ascaled_graph_loss"
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
=	variables
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_

0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#  (2GraphRegularization/total
%:#  (2GraphRegularization/count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:, 2Adam/conv2d_2/kernel/m
 : 2Adam/conv2d_2/bias/m
%:#
??@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:, 2Adam/conv2d_2/kernel/v
 : 2Adam/conv2d_2/bias/v
%:#
??@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
2__inference_GraphRegularization_layer_call_fn_6505
2__inference_GraphRegularization_layer_call_fn_7020
2__inference_GraphRegularization_layer_call_fn_7056
2__inference_GraphRegularization_layer_call_fn_6791?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7405
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7754
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6866
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6941?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_5682NL_nbr_0_imageNL_nbr_0_weightNL_nbr_1_imageNL_nbr_1_weightNL_nbr_2_imageNL_nbr_2_weightimage"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_5860
)__inference_sequential_layer_call_fn_7779
)__inference_sequential_layer_call_fn_7804
)__inference_sequential_layer_call_fn_6039
)__inference_sequential_layer_call_fn_7829
)__inference_sequential_layer_call_fn_7854?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_7898
D__inference_sequential_layer_call_and_return_conditional_losses_7942
D__inference_sequential_layer_call_and_return_conditional_losses_6072
D__inference_sequential_layer_call_and_return_conditional_losses_6105
D__inference_sequential_layer_call_and_return_conditional_losses_7987
D__inference_sequential_layer_call_and_return_conditional_losses_8032?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_graph_loss_layer_call_fn_8039?
???
FullArgSpec(
args ?
jself
jinputs
	jweights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_graph_loss_layer_call_and_return_conditional_losses_8280?
???
FullArgSpec(
args ?
jself
jinputs
	jweights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_6984NL_nbr_0_imageNL_nbr_0_weightNL_nbr_1_imageNL_nbr_1_weightNL_nbr_2_imageNL_nbr_2_weightimage"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_8289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_8300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_max_pooling2d_layer_call_fn_8305
,__inference_max_pooling2d_layer_call_fn_8310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8315
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_1_layer_call_fn_8329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_1_layer_call_fn_8345
.__inference_max_pooling2d_1_layer_call_fn_8350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8355
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8360?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_2_layer_call_fn_8369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8380?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_2_layer_call_fn_8385
.__inference_max_pooling2d_2_layer_call_fn_8390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8395
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_8405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_8411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_8420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_8431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_8440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_8451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6866?"#$%&'()*+?????
???
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????
p 
? "3?0
?
0?????????
?
?	
1/0 ?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_6941?"#$%&'()*+?????
???
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????
p
? "3?0
?
0?????????
?
?	
1/0 ?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7405?"#$%&'()*+?????
???
???
[
NL_nbr_0_imageI?F
inputs/NL_nbr_0_image+???????????????????????????
C
NL_nbr_0_weight0?-
inputs/NL_nbr_0_weight?????????
[
NL_nbr_1_imageI?F
inputs/NL_nbr_1_image+???????????????????????????
C
NL_nbr_1_weight0?-
inputs/NL_nbr_1_weight?????????
[
NL_nbr_2_imageI?F
inputs/NL_nbr_2_image+???????????????????????????
C
NL_nbr_2_weight0?-
inputs/NL_nbr_2_weight?????????
9
image0?-
inputs/image???????????
p 
? "3?0
?
0?????????
?
?	
1/0 ?
M__inference_GraphRegularization_layer_call_and_return_conditional_losses_7754?"#$%&'()*+?????
???
???
[
NL_nbr_0_imageI?F
inputs/NL_nbr_0_image+???????????????????????????
C
NL_nbr_0_weight0?-
inputs/NL_nbr_0_weight?????????
[
NL_nbr_1_imageI?F
inputs/NL_nbr_1_image+???????????????????????????
C
NL_nbr_1_weight0?-
inputs/NL_nbr_1_weight?????????
[
NL_nbr_2_imageI?F
inputs/NL_nbr_2_image+???????????????????????????
C
NL_nbr_2_weight0?-
inputs/NL_nbr_2_weight?????????
9
image0?-
inputs/image???????????
p
? "3?0
?
0?????????
?
?	
1/0 ?
2__inference_GraphRegularization_layer_call_fn_6505?"#$%&'()*+?????
???
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????
p 
? "???????????
2__inference_GraphRegularization_layer_call_fn_6791?"#$%&'()*+?????
???
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????
p
? "???????????
2__inference_GraphRegularization_layer_call_fn_7020?"#$%&'()*+?????
???
???
[
NL_nbr_0_imageI?F
inputs/NL_nbr_0_image+???????????????????????????
C
NL_nbr_0_weight0?-
inputs/NL_nbr_0_weight?????????
[
NL_nbr_1_imageI?F
inputs/NL_nbr_1_image+???????????????????????????
C
NL_nbr_1_weight0?-
inputs/NL_nbr_1_weight?????????
[
NL_nbr_2_imageI?F
inputs/NL_nbr_2_image+???????????????????????????
C
NL_nbr_2_weight0?-
inputs/NL_nbr_2_weight?????????
9
image0?-
inputs/image???????????
p 
? "???????????
2__inference_GraphRegularization_layer_call_fn_7056?"#$%&'()*+?????
???
???
[
NL_nbr_0_imageI?F
inputs/NL_nbr_0_image+???????????????????????????
C
NL_nbr_0_weight0?-
inputs/NL_nbr_0_weight?????????
[
NL_nbr_1_imageI?F
inputs/NL_nbr_1_image+???????????????????????????
C
NL_nbr_1_weight0?-
inputs/NL_nbr_1_weight?????????
[
NL_nbr_2_imageI?F
inputs/NL_nbr_2_image+???????????????????????????
C
NL_nbr_2_weight0?-
inputs/NL_nbr_2_weight?????????
9
image0?-
inputs/image???????????
p
? "???????????
__inference__wrapped_model_5682?"#$%&'()*+?????
???
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????
? "3?0
.
output_1"?
output_1??????????
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8340p$%9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_conv2d_1_layer_call_fn_8329c$%9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8380l&'7?4
-?*
(?%
inputs?????????KK
? "-?*
#? 
0?????????KK 
? ?
'__inference_conv2d_2_layer_call_fn_8369_&'7?4
-?*
(?%
inputs?????????KK
? " ??????????KK ?
@__inference_conv2d_layer_call_and_return_conditional_losses_8300p"#9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_conv2d_layer_call_fn_8289c"#9?6
/?,
*?'
inputs???????????
? ""?????????????
A__inference_dense_1_layer_call_and_return_conditional_losses_8451\*+/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? y
&__inference_dense_1_layer_call_fn_8440O*+/?,
%?"
 ?
inputs?????????@
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_8431^()1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? y
$__inference_dense_layer_call_fn_8420Q()1?.
'?$
"?
inputs???????????
? "??????????@?
A__inference_flatten_layer_call_and_return_conditional_losses_8411b7?4
-?*
(?%
inputs?????????&& 
? "'?$
?
0???????????
? 
&__inference_flatten_layer_call_fn_8405U7?4
-?*
(?%
inputs?????????&& 
? "?????????????
D__inference_graph_loss_layer_call_and_return_conditional_losses_8280???
x?u
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????

 
? "?

?
0 
? ?
)__inference_graph_loss_layer_call_fn_8039???
x?u
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????

 
? "? ?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8355?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8360j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????KK
? ?
.__inference_max_pooling2d_1_layer_call_fn_8345?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_1_layer_call_fn_8350]9?6
/?,
*?'
inputs???????????
? " ??????????KK?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8395?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8400h7?4
-?*
(?%
inputs?????????KK 
? "-?*
#? 
0?????????&& 
? ?
.__inference_max_pooling2d_2_layer_call_fn_8385?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_2_layer_call_fn_8390[7?4
-?*
(?%
inputs?????????KK 
? " ??????????&& ?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8315?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8320l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_max_pooling2d_layer_call_fn_8305?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
,__inference_max_pooling2d_layer_call_fn_8310_9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_sequential_layer_call_and_return_conditional_losses_6072u
"#$%&'()*+@?=
6?3
)?&
image???????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_6105u
"#$%&'()*+@?=
6?3
)?&
image???????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_7898v
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_7942v
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_7987?
"#$%&'()*+U?R
K?H
>?;
9
image0?-
inputs/image???????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8032?
"#$%&'()*+U?R
K?H
>?;
9
image0?-
inputs/image???????????
p

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_5860h
"#$%&'()*+@?=
6?3
)?&
image???????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_6039h
"#$%&'()*+@?=
6?3
)?&
image???????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_7779i
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_7804i
"#$%&'()*+A?>
7?4
*?'
inputs???????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_7829}
"#$%&'()*+U?R
K?H
>?;
9
image0?-
inputs/image???????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_7854}
"#$%&'()*+U?R
K?H
>?;
9
image0?-
inputs/image???????????
p

 
? "???????????
"__inference_signature_wrapper_6984?"#$%&'()*+?????
? 
???
T
NL_nbr_0_imageB??
NL_nbr_0_image+???????????????????????????
<
NL_nbr_0_weight)?&
NL_nbr_0_weight?????????
T
NL_nbr_1_imageB??
NL_nbr_1_image+???????????????????????????
<
NL_nbr_1_weight)?&
NL_nbr_1_weight?????????
T
NL_nbr_2_imageB??
NL_nbr_2_image+???????????????????????????
<
NL_nbr_2_weight)?&
NL_nbr_2_weight?????????
2
image)?&
image???????????"3?0
.
output_1"?
output_1?????????