/* 부적합 양식 변환기 — 웹앱 설치용 서비스 워커
 * - 사이트 파일(index.html, model.json 등)은 항상 인터넷에서 먼저 받고, 끊겼을 때만 저장본을 씀 → 수정한 파일이 바로 반영됨
 * - 구글시트·Apps Script(관리대장 서버) 요청은 건드리지 않음 (항상 실시간)
 */
const CACHE = 'nc-app-v4';   // 아이콘·로고를 바꾸면 숫자를 올려야 새 그림으로 바뀜
const CORE = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './apple-touch-icon.png', './nb-icon.svg', './ninebell-logo.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => Promise.all(CORE.map(u => c.add(u).catch(() => {})))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;   // 다른 사이트(구글시트·서버·라이브러리)는 그대로
  if (/\.(png|svg)$/.test(url.pathname)) {   // 아이콘·로고: 저장본 먼저
    e.respondWith(caches.match(req).then(r => r || fetch(req)));
    return;
  }
  e.respondWith(   // 나머지: 인터넷 먼저 → 실패하면 저장본
    fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true }).then(r => r || caches.match('./index.html')))
  );
});
