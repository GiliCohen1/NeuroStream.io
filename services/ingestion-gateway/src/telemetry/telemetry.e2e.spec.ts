import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { Observable, of, throwError } from 'rxjs';
import request from 'supertest';
import { AppModule } from '../app.module';
import { configureApp } from '../configure-app';
import { KAFKA_CLIENT, RAW_SIGNALS_TOPIC } from './telemetry.constants';

/**
 * Runs the REAL gateway (routing, validation pipe, controller, service, exception filter)
 * over real HTTP. Only the Kafka client is replaced, by an in-memory fake that records
 * exactly what would have been produced to the broker.
 */
class FakeKafka {
  sent: { topic: string; key: string; value: string }[] = [];
  failNext = false;
  connect = async () => undefined;
  close = async () => undefined;
  emit(topic: string, msg: { key: string; value: string }): Observable<unknown> {
    if (this.failNext) return throwError(() => new Error('broker down'));
    this.sent.push({ topic, ...msg });
    return of([{ topicName: topic, partition: 0 }]);
  }
}

const validChunk = (over: Record<string, unknown> = {}) => ({
  patientId: 'patient-001',
  timestamp: 1700000000000,
  signalVector: Array.from({ length: 178 }, (_, i) => Math.sin(i / 5)),
  ...over,
});

describe('POST /api/v1/telemetry/stream (e2e, fake Kafka)', () => {
  let app: INestApplication;
  const kafka = new FakeKafka();

  beforeAll(async () => {
    const mod = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(KAFKA_CLIENT)
      .useValue(kafka)
      .compile();
    app = mod.createNestApplication();
    configureApp(app);
    await app.init();
  });
  afterAll(() => app.close());
  beforeEach(() => {
    kafka.sent = [];
    kafka.failNext = false;
  });

  it('accepts a valid chunk with 202 and publishes it to neuro-raw-signals keyed by patientId', async () => {
    const body = validChunk();
    const res = await request(app.getHttpServer()).post('/api/v1/telemetry/stream').send(body);

    expect(res.status).toBe(202);
    expect(res.body).toEqual({ accepted: true, patientId: 'patient-001', samples: 178 });
    expect(kafka.sent).toHaveLength(1);
    expect(kafka.sent[0].topic).toBe(RAW_SIGNALS_TOPIC);
    expect(kafka.sent[0].topic).toBe('neuro-raw-signals');
    expect(kafka.sent[0].key).toBe('patient-001');
    expect(JSON.parse(kafka.sent[0].value)).toEqual(body); // the worker receives exactly the payload
  });

  it.each([
    ['missing patientId', { patientId: undefined }],
    ['empty patientId', { patientId: '' }],
    ['non-positive timestamp', { timestamp: 0 }],
    ['signal too short', { signalVector: [1, 2, 3] }],
    ['non-numeric samples', { signalVector: Array(32).fill('x') }],
    ['unknown extra field', { hack: true }],
  ])('rejects %s with 400 and publishes nothing', async (_name, over) => {
    const res = await request(app.getHttpServer()).post('/api/v1/telemetry/stream').send(validChunk(over));
    expect(res.status).toBe(400);
    expect(kafka.sent).toHaveLength(0);
  });

  it('returns 503 (no internals leaked) when Kafka is unavailable', async () => {
    kafka.failNext = true;
    const res = await request(app.getHttpServer()).post('/api/v1/telemetry/stream').send(validChunk());
    expect(res.status).toBe(503);
    expect(JSON.stringify(res.body)).not.toMatch(/broker down/);
  });

  it('serves a health check', async () => {
    const res = await request(app.getHttpServer()).get('/health');
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: 'ok' });
  });
});
