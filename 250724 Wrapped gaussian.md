# Wrapped Gaussian
250724
wrapped gaussian을 통해서 초기 분포를 형성하며, Order parameter와 Daido parameter(1차 및 2차 모멘트)를 구해 보려고 한다.

초기 조건은 다음과 같이 구성되어 있다.
$$
f_\sigma = \eta_\sigma f_a + (1-\eta_\sigma) f_b
$$
$f_a,f_b$는 각각 0과 pi에 있는 임의의 unimodal한 분포이며 분산은 서로 같다고 가정을 하였으며, $\eta$는 0 혹은 $\pi$중에서 어느 위치에 많이 분포하는 가를 의미한다. $f_a,f_b$의 경우 현제는 가우시안 분포를 사용할 것이다.

## Wrapped gaussian distribution n-th moment

$$
{\displaystyle f_{WN}(\theta ;\mu ,\sigma )={\frac {1}{\sigma {\sqrt {2\pi }}}}\sum _{k=-\infty }^{\infty }\exp \left[{\frac {-(\theta -\mu +2\pi k)^{2}}{2\sigma ^{2}}}\right],}
$$
@wikipedia
periodic boundary condition에서의 가우시안 분포는 위의 식과 같이 구성이 된다. 

또한 여기의 moment는 다음과 같다.
$$
{\displaystyle \langle z^{n}\rangle =\int _{\Gamma }e^{in\theta }\,f_{WN}(\theta ;\mu ,\sigma )\,d\theta =e^{in\mu -n^{2}\sigma ^{2}/2}.}
$$
우리가 구하려고 하는 order parameter와 Daido order parameter의 경우 각각 1차 및 2차 moment이며, 위상차가 $pi$만큼 차이나는 wrapped normal distribution이 합쳐진 분포이다.

$$
{\displaystyle \langle z^{n}\rangle =\int _{\Gamma }e^{in\theta }\,[\eta_\sigma f_{WN}(\theta ;\mu ,\sigma ) + (1=\eta_\sigma)f_{WN}(\theta ;\mu + \pi ,\sigma )]\,d\theta =\eta_\sigma e^{in\mu -n^{2}\sigma ^{2}/2}} +(1-\eta_\sigma) e^{in\mu+in\pi -n^{2}\sigma ^{2}/2}
$$

여기서 $\mu=0$으로 설정하는 경우 n-th order parameter는 다음과 같이 설정이 가능하다.
$$
R_n =|\langle z^{n}\rangle|  = \eta_\sigma e^{-n^{2}\sigma ^{2}/2} +(-1)^n(1-\eta_\sigma) e^{-n^{2}\sigma ^{2}/2}
$$
Order parameter는 다음과 같이 표현 가능하며
$$
R_1 = |1-2\eta_\sigma| e^{-\sigma ^{2}/2},
$$
Daido order parameter의 경우 다음과 같이 표현이 가능하다:
$$
R_2 = e^{-2\sigma ^{2}},
$$
 
이제 nordmal distirubtion과 standard deviation의 관계를 나타내면, 

$$
\sigma = \sqrt{-\frac{1}{2} \ln R_2}
$$
또한 각각의 분포에 대한 정보와, Order parameter와 Daido order parameter을 사용한다면
$$
R_2^{1/4} = e^{-\sigma^2/2}
$$
$$
|1-2\eta_\sigma| = \frac{R_1}{R_2^{1/4}}
$$

---
# OA ansatz 관련된 정리


OA anstaz에서 기본적인 전제는 다음과 같은 관계를 가진다
$$
a_n = a^n
$$
$$
a_2 = a_1^2
$$
$$
a_n =\langle z^{n}\rangle  = e^{-n^{2}\sigma ^{2}/2}
$$
$$
A_n = \eta e^{-n^{2}\sigma ^{2}/2} + (1-\eta) e^{-in\pi}e^{-n^{2}\sigma ^{2}/2}
$$

$$
a_1 =\langle z^{1}\rangle  = e^{-1\sigma ^{2}/2}, a_2 =\langle z^{2}\rangle  = e^{-4\sigma ^{2}/2}
$$
$$
a_2 = e^{-2\sigma ^{2}} = (e^{-\sigma ^{2}/2})^4
$$
$$
a_3 = e^{-9\sigma ^{2}/2} = (e^{-\sigma ^{2}/2})^9
$$

$$
a^{(n+2)} =  e^{-(n+2)^{2}\sigma ^{2}/2} =  e^{-(n^2+4n+4)\sigma ^{2}/2} = e^{-n^2\sigma ^{2}/2} e^{-(2n + 2)\sigma ^{2}}
$$
$$
a^{(n+2)} = a^{(n)} e^{-(4n + 4)\sigma ^{2}/2} = a^{(n)} a_1^{(4n+4)}
$$
$$
a^{(n)} =  e^{-n^2 \sigma ^{2}/2} = a_1^{n^2}
$$
$$
a^{(n-2)} =  e^{-(n-2)^{2}\sigma ^{2}/2} =  e^{-(n^2-4n+4)\sigma ^{2}/2} = e^{-(n^2)\sigma ^{2}/2} e^{-(-2n + 2)\sigma ^{2}}
$$
$$
a^{(n-2)} = a^{(n)} e^{-(4n + 4)\sigma ^{2}/2} = a^{(n)} a_1^{(-4n+4)}
$$

OA anstaz의 식은 다음과 같다.

$$
 \frac{\partial f_\sigma}{\partial t}+\frac{\partial\left(v_\sigma f_\sigma\right)}{\partial\theta}=0=\sum_{n=1}^{\infty}{\frac{1}{2\pi}\left(\dot{a_\sigma^{\left(n\right)}}+n\cdot\ ia_\sigma^{\left(n\right)}\omega+\frac{nK}{2}\ (H_\sigma e^{-i\alpha}a_\sigma^{\left(n+2\right)}-H_\sigma^\ast e^{i\alpha}a_\sigma^{\left(n-2\right)})\right)e^{ni\theta}}+c.c.
$$

$$
\sum_{n=1}^{\infty}{\frac{1}{2\pi}\left(\dot{a_\sigma^{\left(n\right)}}+n\cdot\ ia_\sigma^{\left(n\right)}\omega+\frac{nK}{2}\ (H_\sigma e^{-i\alpha}a_\sigma^{\left(n+2\right)}-H_\sigma^\ast e^{i\alpha}a_\sigma^{\left(n-2\right)})\right)e^{ni\theta}}+c.c.
$$

$$
\dot{a^{(n)}} =  n^2 a_1^{n^2 - 1}\dot{a_1}
$$


$$
\sum_{n=1}^{\infty}{\frac{1}{2\pi}\left(n^2 a_1^{n^2 - 1}\dot{a_1}+n\cdot\ ia_1^{n^2}\omega+\frac{nK}{2}\ (H_\sigma e^{-i\alpha}a_1^{\left(n+2\right)^2}-H_\sigma^\ast e^{i\alpha}a_\sigma^{\left(n-2\right)^2})\right)e^{ni\theta}}+c.c.
$$

$$
\sum_{n=1}^{\infty}{\frac{ n^2a_1^{n^2 - 1}}{2\pi}\left(\dot{a_1}+\frac{1}{n} ia_1\omega+\frac{K}{2n}\ (H_\sigma e^{-i\alpha}a_1^{\left(4n+5\right)}-H_\sigma^\ast e^{i\alpha}a_\sigma^{\left(4n-3\right)})\right)e^{ni\theta}}+c.c.
$$

다음과 같이 정리가 되며 OA ansatz와 달리 기본 분포가 gaussian distribution이라고 가정할떄 OA ansatz는 하나의 phase reduction equation으로 정리되지 않는다