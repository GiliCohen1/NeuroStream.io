/* Site chrome: header state, mobile menu, scroll-spy, reveal-on-scroll, count-up. No dependencies. */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // Header gains a hairline once the page scrolls.
  var header = $('.site-header');
  function onScroll() { header.classList.toggle('scrolled', window.scrollY > 8); }
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  // Mobile menu (disclosure pattern).
  var btn = $('#menubtn'), menu = $('#menu');
  btn.addEventListener('click', function () {
    var open = menu.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(open));
  });
  $$('#menu a').forEach(function (a) {
    a.addEventListener('click', function () { menu.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); });
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && menu.classList.contains('open')) { menu.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); btn.focus(); }
  });

  if (!('IntersectionObserver' in window)) { $$('.reveal').forEach(function (el) { el.classList.add('in'); }); return; }

  // Scroll-spy for the primary navigation.
  var links = {};
  $$('#menu a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
  var spy = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      Object.keys(links).forEach(function (k) { links[k].removeAttribute('aria-current'); });
      if (links[en.target.id]) links[en.target.id].setAttribute('aria-current', 'true');
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  Object.keys(links).forEach(function (id) { var s = document.getElementById(id); if (s) spy.observe(s); });

  // Reveal on scroll.
  var rev = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); rev.unobserve(en.target); } });
  }, { rootMargin: '0px 0px -8% 0px', threshold: .08 });
  $$('.reveal').forEach(function (el) { rev.observe(el); });

  // Count-up for headline figures.
  function countUp(el) {
    var target = parseFloat(el.getAttribute('data-count')), dec = +(el.getAttribute('data-dec') || 0);
    var suffix = el.getAttribute('data-suffix') || '';
    if (reduce) { el.textContent = target.toFixed(dec) + suffix; return; }
    var t0 = null, dur = 1100;
    function step(t) {
      if (t0 === null) t0 = t;
      var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * e).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var cu = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) { if (en.isIntersecting) { countUp(en.target); cu.unobserve(en.target); } });
  }, { threshold: .6 });
  $$('[data-count]').forEach(function (el) { cu.observe(el); });
})();
