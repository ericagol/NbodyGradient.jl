# Replacement for deprecated functions:

function linearspace(a,b,n)
if VERSION >= v"0.7"
  return range(a,stop=b,length=n)
else
  return linspace(a,b,n)
end
return
end

function logarithmspace(a,b,n)
if VERSION >= v"0.7"
  return 10.0 .^range(a,stop=b,length=n)
else
  return logspace(a,b,n)
end
return
end
