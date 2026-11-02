import 'reflect-metadata';
import { plainToInstance } from 'class-transformer';
import { validate } from 'class-validator';
import { TelemetryChunkDto } from './telemetry.dto';

const make = (o: Record<string, unknown>) =>
  validate(plainToInstance(TelemetryChunkDto, { patientId: 'p1', timestamp: 1700000000000, signalVector: Array(32).fill(1), ...o }));

describe('TelemetryChunkDto', () => {
  it('accepts a valid chunk', async () => expect(await make({})).toHaveLength(0));
  it('rejects empty patientId', async () => expect(await make({ patientId: '' })).not.toHaveLength(0));
  it('rejects too-short vectors', async () => expect(await make({ signalVector: [1, 2] })).not.toHaveLength(0));
  it('rejects non-numeric samples', async () => expect(await make({ signalVector: Array(32).fill('x') })).not.toHaveLength(0));
  it('rejects non-positive timestamp', async () => expect(await make({ timestamp: -1 })).not.toHaveLength(0));
});
