function [sys,x0,str,ts,simStateCompliance] = ADRC_TD(t,x,u,flag,para)
switch flag
  % Initialization %
  case 0
    [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(para);
  % Derivatives %
  case 1
    sys=mdlDerivatives(t,x,u);
  % Update %
  case 2
    sys=mdlUpdate(t,x,u,para);
  % Outputs %
  case 3
    sys=mdlOutputs(t,x,u);
  % GetTimeOfNextVarHit %
  case 4
    sys=mdlGetTimeOfNextVarHit(t,x,u);
  % Terminate %
  case 9
    sys=mdlTerminate(t,x,u);
  % Unexpected flags %
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
end

function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(para)

sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 2;
sizes.NumOutputs     = 2;
sizes.NumInputs      = 1;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);
% initialize the initial conditions
x0  = [0 0];  %% NOTE: 赋值时可以空格(行向量)也可以分号(列向量), 但是x是列向量
% str is always an empty matrix
str = [];
% initialize the array of sample times
ts  = [para.h 0];

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'DisallowSimState' < Error out when saving or restoring the model sim state
simStateCompliance = 'UnknownSimState';


function sys=mdlDerivatives(t,x,u)

sys = [];

function sys=mdlUpdate(t,x,u,para)
h=para.h; % step

x1=x(1)+h*x(2);
fh=fhan(x(1)-u,x(2),para.r0,para.h0);
x2=x(2)+h*fh;
sys = [x1 x2];

function fh=fhan(x1,x2,r0,h0) % fhan function for adrc
d=r0*h0^2;
a0=h0*x2;
y=x1+a0;
a1=sqrt(d*(d+8*abs(y)));
a2=a0+sign(y)*(a1-d)/2;
sy=(sign(y+d)-sign(y-d))/2;
a=(a0+y-a2)*sy+a2;
sa=(sign(a+d)-sign(a-d))/2;
fh=-r0*(a/d-sign(a))*sa-r0*sign(a);

function sys=mdlOutputs(t,x,u)
sys = x;

function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

function sys=mdlTerminate(t,x,u)

sys = [];
