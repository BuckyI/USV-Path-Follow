function [sys,x0,str,ts,simStateCompliance] = ADRC_ESO(t,x,u,flag,para)
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
sizes.NumDiscStates  = 3;
sizes.NumOutputs     = 3;  % NOTE: z1, z2, z3
sizes.NumInputs      = 2;  % NOTE: y, u
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);
% initialize the initial conditions
x0  = [0 0 0];  %% NOTE: 赋值时可以空格(行向量)也可以分号(列向量), 但是x是列向量
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
%y=u(1) u=u(2)
e=x(1)-u(1);
z1=x(1)+h*x(2)-para.beta01*e;
z2=x(2)+h*(x(3)+para.b0*u(2))-para.beta02*fal(e,0.5,h);
z3=x(3)-para.beta03*fal(e,0.25,h);
sys = [z1 z2 z3];

function fa=fal(x,a,delta) % fal function for adrc
if abs(x)<=delta
    fa=x/(delta^(1-a));
else
    fa=sign(x)*abs(x)^a;
end


function sys=mdlOutputs(t,x,u)
sys = x;

function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

function sys=mdlTerminate(t,x,u)

sys = [];
